# Export model blocks to ONNX.
# Author: paspf
#
# Usage:
# Call this script from cli to export the desired model block.
#
# Exported model blocks:
#   image-encoder  : trunk + neck + conv_s0/conv_s1 projection for high-res features.
#                    Outputs: high_res_feat_0, high_res_feat_1, image_embedding,
#                             vision_pos_enc (all at the final spatial resolution used by
#                             the SAM heads and memory attention).
#   sam-heads      : PromptEncoder + MaskDecoder (combined).  Handles multimask
#                    selection internally and returns the best low-/high-res mask,
#                    the object pointer and object score logits.
#   memory-encoder : Encodes a (pix_feat, sigmoid-scaled mask) pair into the
#                    compact memory representation (maskmem_features + maskmem_pos_enc).
#   memory-attention: Fuses current vision features with the accumulated memory bank.
#                    Supports a dynamic memory length via a dynamic ONNX axis.

import onnx
from pathlib import Path
from nanosam2.datasets.containers import ModelSource
import torch
import torch.nn as nn
import torch.nn.functional as F
from nanosam2.sam2.build_sam import build_sam2_video_predictor
import numpy as np
import onnxruntime as ort
import onnxsim


# ---------------------------------------------------------------------------
# ONNX wrapper modules
# ---------------------------------------------------------------------------

class ImageEncoderONNXWrapper(nn.Module):
    """
    Wraps the full image-encoder forward pass (backbone trunk + FPN neck) and
    applies conv_s0 / conv_s1 so that the high-resolution feature maps are
    already in the form expected by the SAM mask decoder.

    Outputs (6 tensors):
        high_res_feat_0  – (B, C/8, H/4,  W/4)   e.g. (1,  32, 128, 128)
        high_res_feat_1  – (B, C/4, H/8,  W/8)   e.g. (1,  64,  64,  64)
        image_embedding  – (B, C,   H/16, W/16)  e.g. (1, 256,  32,  32)
        vision_pos_enc   – (B, C,   H/16, W/16)  same spatial as image_embedding
        backbone_fpn_raw_0 – (B, C, H/4,  W/4)  raw 256-ch feature (pre conv_s0)
        backbone_fpn_raw_1 – (B, C, H/8,  W/8)  raw 256-ch feature (pre conv_s1)
    """

    def __init__(self, predictor):
        super().__init__()
        self.image_encoder = predictor.image_encoder
        self.use_high_res = predictor.use_high_res_features_in_sam
        if self.use_high_res:
            self.conv_s0 = predictor.sam_mask_decoder.conv_s0
            self.conv_s1 = predictor.sam_mask_decoder.conv_s1

    def forward(self, x: torch.Tensor):
        out = self.image_encoder(x)
        fpn = out["backbone_fpn"]   # list of 3 tensors after scalp
        pos = out["vision_pos_enc"] # list of 3 positional encodings

        raw_fpn0 = fpn[0]  # (B, 256, H/4,  W/4)
        raw_fpn1 = fpn[1]  # (B, 256, H/8,  W/8)

        if self.use_high_res:
            hr0 = self.conv_s0(fpn[0])  # (B, C/8, H/4,  W/4)
            hr1 = self.conv_s1(fpn[1])  # (B, C/4, H/8,  W/8)
        else:
            hr0 = fpn[0]
            hr1 = fpn[1]

        image_embedding = fpn[2]         # (B, C,   H/16, W/16)
        vision_pos_enc  = pos[2]         # (B, C,   H/16, W/16)

        return hr0, hr1, image_embedding, vision_pos_enc, raw_fpn0, raw_fpn1


class SAMHeadsONNXWrapper(nn.Module):
    """
    Combined PromptEncoder + MaskDecoder ONNX wrapper.

    Always runs with multimask_output=True and selects the best mask by IoU
    score, then computes the object pointer.  This covers both the
    initial-conditioning-frame case (user click with ≤1 point) and the
    ongoing tracking case (empty prompt / null mask inputs).

    Inputs:
        image_embedding  – (B, C,   H/16, W/16)
        high_res_feat_0  – (B, C/8, H/4,  W/4)
        high_res_feat_1  – (B, C/4, H/8,  W/8)
        point_coords     – (B, N, 2)  float32  absolute pixel coords
        point_labels     – (B, N)     int32    1=fg, 0=bg, -1=pad/empty

    Outputs:
        low_res_masks        – (B, 1, H/4,  W/4)
        high_res_masks       – (B, 1, H,    W)
        iou_predictions      – (B, 3)  IoU scores for the 3 multimask outputs
        obj_ptr              – (B, C)  object pointer for memory bank
        object_score_logits  – (B, 1)  raw logits for object presence
    """

    def __init__(self, predictor):
        super().__init__()
        self.sam_prompt_encoder = predictor.sam_prompt_encoder
        self.sam_mask_decoder   = predictor.sam_mask_decoder
        self.obj_ptr_proj       = predictor.obj_ptr_proj
        self.pred_obj_scores    = predictor.pred_obj_scores
        self.fixed_no_obj_ptr   = predictor.fixed_no_obj_ptr
        self.soft_no_obj_ptr    = predictor.soft_no_obj_ptr
        self.image_size         = predictor.image_size
        self.use_high_res       = predictor.use_high_res_features_in_sam
        if self.pred_obj_scores and hasattr(predictor, "no_obj_ptr"):
            self.no_obj_ptr = nn.Parameter(predictor.no_obj_ptr.data.clone())

    def forward(
        self,
        image_embedding:  torch.Tensor,
        high_res_feat_0:  torch.Tensor,
        high_res_feat_1:  torch.Tensor,
        point_coords:     torch.Tensor,
        point_labels:     torch.Tensor,
    ):
        # --- prompt encoding ------------------------------------------------
        sparse_emb, dense_emb = self.sam_prompt_encoder(
            points=(point_coords, point_labels),
            boxes=None,
            masks=None,
        )

        # --- mask decoding (always multimask) --------------------------------
        high_res_features = [high_res_feat_0, high_res_feat_1] if self.use_high_res else None
        low_res_multimasks, ious, sam_tokens_out, object_score_logits = self.sam_mask_decoder(
            image_embeddings=image_embedding,
            image_pe=self.sam_prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_emb,
            dense_prompt_embeddings=dense_emb,
            multimask_output=True,  # always output 3 masks; select best below
            repeat_image=False,
            high_res_features=high_res_features,
        )

        # --- upscale masks --------------------------------------------------
        low_res_multimasks = low_res_multimasks.float()
        high_res_multimasks = F.interpolate(
            low_res_multimasks,
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        )

        # --- select best mask by IoU ----------------------------------------
        B = image_embedding.size(0)
        best_iou_inds = torch.argmax(ious, dim=-1)
        batch_inds    = torch.arange(B, device=image_embedding.device)
        low_res_masks  = low_res_multimasks[batch_inds,  best_iou_inds].unsqueeze(1)
        high_res_masks = high_res_multimasks[batch_inds, best_iou_inds].unsqueeze(1)

        # sam_tokens_out: (B, 3, C) when multimask=True
        sam_output_token = sam_tokens_out[batch_inds, best_iou_inds]  # (B, C)

        # --- object pointer -------------------------------------------------
        obj_ptr = self.obj_ptr_proj(sam_output_token)
        if self.pred_obj_scores:
            is_obj_appearing       = object_score_logits > 0
            lambda_is_obj          = is_obj_appearing.float()
            if self.fixed_no_obj_ptr:
                obj_ptr = lambda_is_obj * obj_ptr
            obj_ptr = obj_ptr + (1.0 - lambda_is_obj) * self.no_obj_ptr

        return low_res_masks, high_res_masks, ious, obj_ptr, object_score_logits


class MemoryEncoderONNXWrapper(nn.Module):
    """
    Wraps the memory encoder.

    Inputs:
        pix_feat – (B, C, H/16, W/16)  raw backbone feature (NOT memory-attended)
        mask     – (B, 1, H, W)        sigmoid-scaled mask logits
                   (i.e. sigmoid(high_res_mask) * scale + bias, already computed
                    by the caller before passing here; skip_mask_sigmoid=True)

    Outputs:
        maskmem_features – (B, mem_dim, H/16, W/16)
        maskmem_pos_enc  – (B, mem_dim, H/16, W/16)
    """

    def __init__(self, predictor):
        super().__init__()
        self.memory_encoder = predictor.memory_encoder

    def forward(self, pix_feat: torch.Tensor, mask: torch.Tensor):
        out = self.memory_encoder(pix_feat, mask, skip_mask_sigmoid=True)
        return out["vision_features"], out["vision_pos_enc"][0]


class MemoryAttentionONNXWrapper(nn.Module):
    """
    Wraps the memory attention module with a fixed (tensor-based) interface.

    Inputs:
        curr        – (HW, B, C)       current vision features (seq-first)
        curr_pos    – (HW, B, C)       positional encoding for current features
        memory      – (M,  B, mem_dim) concatenated memory bank  (dynamic M)
        memory_pos  – (M,  B, mem_dim) positional encoding for memory bank

    Output:
        attended    – (HW, B, C)       memory-conditioned features (seq-first)
    """

    def __init__(self, predictor):
        super().__init__()
        self.memory_attention = predictor.memory_attention
        hw16 = getattr(predictor, '_export_hw16', None)
        _make_rope_onnx_safe(self.memory_attention, hw16=hw16)

    def forward(
        self,
        curr:       torch.Tensor,
        curr_pos:   torch.Tensor,
        memory:     torch.Tensor,
        memory_pos: torch.Tensor,
    ) -> torch.Tensor:
        return self.memory_attention(
            curr=curr,
            memory=memory,
            curr_pos=curr_pos,
            memory_pos=memory_pos,
            num_obj_ptr_tokens=0,
        )


# ---------------------------------------------------------------------------
# RoPE helper
# ---------------------------------------------------------------------------

def _make_rope_onnx_safe(module: nn.Module, hw16: int = None) -> None:
    """
    Patch all RoPEAttention layers inside *module* so they use real arithmetic
    instead of complex ops, making them compatible with the dynamo ONNX exporter.

    Complex-number operations (view_as_complex, unsqueeze on complex tensors, etc.)
    have no ONNX mapping. We replace them with equivalent real multiply-add ops.

    hw16: if given, first resize freqs_cis to match the export resolution
          (e.g. hiera at 1024×1024 needs 4096 tokens, not the checkpoint default).
    """
    import math, types
    import torch.nn.functional as F
    from nanosam2.sam2.modeling.sam.transformer import RoPEAttention

    # Maximum number of memory frames we ever concatenate.  The cos/sin buffers
    # for cross-attention layers (rope_k_repeat=True) are pre-tiled to this
    # many repetitions so that a *dynamic* slice [:N] at runtime is all that is
    # needed — no repeat op appears in the ONNX graph.
    MAX_MEM_TILES = 8

    def _rotate_real(x: torch.Tensor, cos: torch.Tensor,
                     sin: torch.Tensor) -> torch.Tensor:
        """Real-arithmetic RoPE rotation.  x: (B, heads, N, D).

        cos/sin may be longer than N (pre-tiled).  We always slice to the
        actual sequence length via a dynamic ONNX Slice node — no repeat
        needed at runtime.
        """
        xr = x.float().reshape(*x.shape[:-1], -1, 2)  # (..., N, D//2, 2)
        N = xr.shape[-3]                               # actual seq len (may be symbolic)
        n_lead = x.ndim - 2
        c = cos.view(*([1] * n_lead), *cos.shape)[..., :N, :]  # (1,1,N,D//2)
        s = sin.view(*([1] * n_lead), *sin.shape)[..., :N, :]
        out_r = xr[..., 0] * c - xr[..., 1] * s
        out_i = xr[..., 0] * s + xr[..., 1] * c
        return torch.stack([out_r, out_i], dim=-1).flatten(-2).type_as(x)

    for child in module.modules():
        if not isinstance(child, RoPEAttention):
            continue
        # Resize freqs_cis if the stored size doesn't match the export resolution
        if hw16 is not None and child.freqs_cis.shape[0] != hw16:
            side = int(math.isqrt(hw16))
            child.freqs_cis = child.compute_cis(end_x=side, end_y=side)
        # Extract cos/sin as plain float32 buffers (no complex dtype in graph).
        # For cross-attention layers the buffers are pre-tiled to MAX_MEM_TILES
        # repetitions so that slicing [:N] at runtime handles any memory length.
        fc = child.freqs_cis.cpu().to(torch.complex64)
        cos_base = fc.real.clone().float()   # (N_base, D//2)
        sin_base = fc.imag.clone().float()
        if child.rope_k_repeat:
            cos_buf = cos_base.repeat(MAX_MEM_TILES, 1)  # (MAX_MEM_TILES*N_base, D//2)
            sin_buf = sin_base.repeat(MAX_MEM_TILES, 1)
        else:
            cos_buf = cos_base
            sin_buf = sin_base
        child.register_buffer("_rope_cos", cos_buf)
        child.register_buffer("_rope_sin", sin_buf)

        # Patch forward to use _rotate_real instead of apply_rotary_enc
        def _make_fwd():
            def _fwd(self, q, k, v, num_k_exclude_rope=0):
                q = self.q_proj(q)
                k = self.k_proj(k)
                v = self.v_proj(v)
                q = self._separate_heads(q, self.num_heads)
                k = self._separate_heads(k, self.num_heads)
                v = self._separate_heads(v, self.num_heads)
                num_k_rope = k.size(-2) - num_k_exclude_rope
                q = _rotate_real(q, self._rope_cos, self._rope_sin)
                k_r = _rotate_real(k[:, :, :num_k_rope],
                                   self._rope_cos, self._rope_sin)
                k = torch.cat([k_r, k[:, :, num_k_rope:]], dim=2)
                out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
                out = self._recombine_heads(out)
                return self.out_proj(out)
            return _fwd

        child.forward = types.MethodType(_make_fwd(), child)


def modify_filename(file_path, str_extension="_modified") -> Path:
    """
    Add the string str_extension before the file extension to the name of a file.
    """
    # Check if the input is already a Path object
    if not isinstance(file_path, Path):
        # Create a Path object from the file path string
        path = Path(file_path)
    else:
        path = file_path

    # Create the new file name by adding "modified" before the extension
    new_file_name = path.with_name(f"{path.stem}{str_extension}{path.suffix}")

    return new_file_name

def remove_duplicate_outputs(model_path:str, out_file:str=None) -> onnx.ModelProto:
    """
    Remove duplicate output nodes from an onnx model.
    Nodes are considered as duplicated if they have the same name and output shape.
    """
    model = onnx.load(model_path)

    # Get the current output nodes
    original_outputs = model.graph.output

    # Dictionary to track unique outputs
    unique_outputs = {}
    new_outputs = []
    to_remove = []

    # Iterate through the output nodes
    for output in original_outputs:
        # Create a key based on the output name and shape
        shape = tuple(dim.dim_value for dim in output.type.tensor_type.shape.dim)
        key = (output.name, shape)

        # Check if this key already exists
        if key not in unique_outputs:
            unique_outputs[key] = output
            new_outputs.append(output)
        else:
            print(f"Duplicate output found: {output.name} with shape: {shape}. Removing it.")
            to_remove.append(output)

    # Update the model's output nodes
    for o in to_remove:
        model.graph.output.remove(o)

    if out_file is not None:
        onnx.save(model, out_file)
        print(f"Modified model saved to {out_file}")
    return {"onnx_model":model, "out_file": out_file}


def analyze_onnx_model(model:onnx.ModelProto):
    """
    Identify the in- and output nodes of a onnx model.
    """
    graph = model.graph

    for input in graph.input:
        output_name = input.name
        output_shape = [dim.dim_value for dim in input.type.tensor_type.shape.dim]
        print(f'Input Name: {output_name}, Shape: {output_shape}')

    for output in graph.output:
        output_name = output.name
        output_shape = [dim.dim_value for dim in output.type.tensor_type.shape.dim]
        print(f'Output Name: {output_name}, Shape: {output_shape}')

def test_onnx_model(model_path:Path, input):
    session = ort.InferenceSession(str(model_path))
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: input})

    print("ONNX model output shapes:")
    for i, output in enumerate(outputs):
        print(f"{i} | shape: {output.shape}")
    return output

def determine_torch_output_shapes(y, id=0):
    """
    Determine the output shapes of the feature map produced by a torch inference.
    """
    if isinstance(y, dict):
        for k,v in y.items():
            if isinstance(v, list) or isinstance(v, tuple):
                determine_torch_output_shapes(v)
            else:
                print(f"{id} | {k} shape: {v.shape}")
                id += 1
    elif isinstance(y, list) or isinstance(y, tuple):
        for k,v in enumerate(y):
            determine_torch_output_shapes(v)
    else:
        print(f"out | - {y.shape}")

def test_torch_model(torch_model:torch.nn, input, silent=False, use_unpack_operator=False) -> bool:
    if use_unpack_operator: y = torch_model(*input)
    else: y = torch_model(input)
    if not silent:
        determine_torch_output_shapes(y)
    return True


def get_block_and_inputs(predictor: torch.nn.Module, block: str, img_shape: list = [3, 512, 512]):
    """Return (torch_model, torch_input, use_unpack_operator, input_names, output_names, dynamic_shapes)."""
    input_names         = ["input"]
    output_names        = None   # None means let torch auto-name
    d_shapes            = None
    use_unpack_operator = True

    H, W = img_shape[1], img_shape[2]
    # Derived spatial dims assuming stride 4/8/16
    h4,  w4  = H // 4,  W // 4   # e.g. 128×128
    h8,  w8  = H // 8,  W // 8   # e.g.  64×64
    h16, w16 = H // 16, W // 16  # e.g.  32×32
    hw16 = h16 * w16              # e.g. 1024

    match block:
        # ------------------------------------------------------------------
        case "nanosam2":
            print("Hint: Full model export is not supported.")
            torch_model = predictor
            torch_input = (torch.randn(1, img_shape[0], H, W),)

        # ------------------------------------------------------------------
        case "image-encoder":
            torch_model  = ImageEncoderONNXWrapper(predictor)
            torch_input  = (torch.randn(1, img_shape[0], H, W),)
            input_names  = ["image"]
            output_names = [
                "high_res_feat_0",    # (1, C/8, H/4,  W/4)  projected by conv_s0
                "high_res_feat_1",    # (1, C/4, H/8,  W/8)  projected by conv_s1
                "image_embedding",    # (1, C,   H/16, W/16) raw backbone feat
                "vision_pos_enc",     # (1, C,   H/16, W/16) position encoding
                "backbone_fpn_raw_0", # (1, C,   H/4,  W/4)  raw 256-ch feat
                "backbone_fpn_raw_1", # (1, C,   H/8,  W/8)  raw 256-ch feat
            ]

        # ------------------------------------------------------------------
        case "image-encoder-trunk":
            torch_model  = predictor.image_encoder.trunk
            torch_input  = (torch.randn(1, img_shape[0], H, W),)
            input_names  = ["image"]

        # ------------------------------------------------------------------
        case "image-encoder-neck":
            # Read backbone output channels from the neck config
            neck = predictor.image_encoder.neck
            ch_list = neck.backbone_channel_list  # e.g. [768,384,192,96] hiera or [512,256,128,64] resnet
            # Spatial sizes: stride 4,8,16,32 from H
            torch_model = neck
            torch_input = [
                torch.randn(1, ch_list[3], h4,          w4         ),
                torch.randn(1, ch_list[2], h8,          w8         ),
                torch.randn(1, ch_list[1], h16,         w16        ),
                torch.randn(1, ch_list[0], h16 // 2,    w16 // 2   ),
            ]
            use_unpack_operator = False

        # ------------------------------------------------------------------
        case "sam-heads":
            torch_model  = SAMHeadsONNXWrapper(predictor)
            n_pts        = 2
            torch_input  = (
                torch.randn(1, 256, h16, w16),
                torch.randn(1, 32,  h4,  w4 ),
                torch.randn(1, 64,  h8,  w8 ),
                torch.randn(1, n_pts, 2).float(),
                torch.randint(0, 2, (1, n_pts)).to(torch.int32),
            )
            input_names  = ["image_embedding", "high_res_feat_0", "high_res_feat_1",
                            "point_coords", "point_labels"]
            output_names = ["low_res_masks", "high_res_masks", "iou_predictions",
                            "obj_ptr", "object_score_logits"]
            _num_pts = torch.export.Dim("num_points", min=1, max=1024)
            d_shapes = (None, None, None, {1: _num_pts}, {1: _num_pts})

        # ------------------------------------------------------------------
        case "mask-decoder":
            # Legacy alias – maps to the same SAMHeadsONNXWrapper
            torch_model  = SAMHeadsONNXWrapper(predictor)
            n_pts        = 2
            torch_input  = (
                torch.randn(1, 256, h16, w16),
                torch.randn(1, 32,  h4,  w4 ),
                torch.randn(1, 64,  h8,  w8 ),
                torch.randn(1, n_pts, 2).float(),
                torch.randint(0, 2, (1, n_pts)).to(torch.int32),
            )
            input_names  = ["image_embedding", "high_res_feat_0", "high_res_feat_1",
                            "point_coords", "point_labels"]
            output_names = ["low_res_masks", "high_res_masks", "iou_predictions",
                            "obj_ptr", "object_score_logits"]
            _num_pts = torch.export.Dim("num_points", min=1, max=1024)
            d_shapes = (None, None, None, {1: _num_pts}, {1: _num_pts})

        # ------------------------------------------------------------------
        case "mask-decoder-transformer":
            torch_model  = predictor.sam_mask_decoder.transformer
            tokens_limit = 8
            objects_limit = 1
            torch_input  = (
                torch.randn(objects_limit, 256, h16, w16),
                torch.randn(objects_limit, 256, h16, w16),
                torch.randn(objects_limit, tokens_limit, 256),
            )
            input_names  = ["src", "pos_src", "tokens"]
            _tok = torch.export.Dim("tokens_dyn_input_num_points", min=1, max=1024)
            d_shapes = (None, None, {1: _tok})

        # ------------------------------------------------------------------
        case "memory-encoder":
            torch_model  = MemoryEncoderONNXWrapper(predictor)
            torch_input  = (
                torch.randn(1, 256, h16, w16),
                torch.randn(1, 1,   H,   W  ),
            )
            input_names  = ["pix_feat", "mask"]
            output_names = ["maskmem_features", "maskmem_pos_enc"]

        # ------------------------------------------------------------------
        case "memory-attention":
            M       = hw16 * 7
            mem_dim = predictor.mem_dim
            # Tag predictor so MemoryAttentionONNXWrapper can pick up hw16
            predictor._export_hw16 = hw16
            torch_model = MemoryAttentionONNXWrapper(predictor)
            del predictor._export_hw16
            torch_input = (
                torch.randn(hw16, 1, 256),
                torch.randn(hw16, 1, 256),
                torch.randn(M,    1, mem_dim),
                torch.randn(M,    1, mem_dim),
            )
            input_names  = ["curr", "curr_pos", "memory", "memory_pos"]
            output_names = ["attended_features"]
            _mem_len = torch.export.Dim("memory_len", min=1, max=hw16 * 8)
            d_shapes = (None, None, {0: _mem_len}, {0: _mem_len})

        # ------------------------------------------------------------------
        case _:
            print(f"Unknown model block: {block}")
            exit()

    return (torch_model, torch_input, use_unpack_operator, input_names, output_names, d_shapes)

def save_model_constants(predictor, model_name: str, out_dir: Path) -> Path:
    """
    Extract the learned constants (embeddings, tpos enc, etc.) from an already-
    loaded PyTorch predictor and save them as a .npz file in *out_dir*.

    The resulting file is loaded automatically by NanoSAM2OnnxPredictor at
    runtime — no PyTorch is required after this step.
    """
    def t(x): return x.detach().cpu().numpy().astype(np.float32)

    no_mem_embed     = t(predictor.no_mem_embed)
    no_mem_pos_enc   = t(predictor.no_mem_pos_enc)
    maskmem_tpos_enc = t(predictor.maskmem_tpos_enc)

    proj_tpos = getattr(predictor, "proj_tpos_enc_in_obj_ptrs", True)
    if proj_tpos and hasattr(predictor, "obj_ptr_tpos_proj"):
        proj = predictor.obj_ptr_tpos_proj
        if hasattr(proj, "weight"):
            tpos_proj_w = t(proj.weight)
            tpos_proj_b = t(proj.bias)
        else:
            # nn.Identity — store as identity matrix
            tpos_proj_w = np.eye(predictor.mem_dim, predictor.hidden_dim,
                                 dtype=np.float32)
            tpos_proj_b = np.zeros(predictor.mem_dim, dtype=np.float32)
    else:
        # No projection — store empty arrays (signals "no projection" to loader)
        tpos_proj_w = np.zeros(0, dtype=np.float32)
        tpos_proj_b = np.zeros(0, dtype=np.float32)

    out_path = out_dir / f"{model_name}-constants.npz"
    np.savez(
        str(out_path),
        no_mem_embed=no_mem_embed,
        no_mem_pos_enc=no_mem_pos_enc,
        maskmem_tpos_enc=maskmem_tpos_enc,
        tpos_proj_w=tpos_proj_w,
        tpos_proj_b=tpos_proj_b,
    )
    return out_path


def export_model_block(m: ModelSource, block: str, out_dir: Path, img_shape: list,
                       use_simplify: bool = False, opset_version: int = 24):
    """
    Export a building block of nanosam2 to ONNX.  Not all blocks can be converted.
    """
    print(f"Exporting Model: {m.name}  block: {block} ...")
    out_dir.mkdir(parents=True, exist_ok=True)
    predictor = build_sam2_video_predictor(m.cfg, m.checkpoint, torch.device("cpu"))
    npz_path = save_model_constants(predictor, m.name, out_dir)
    print(f" - Constants saved to {npz_path}")
    torch_model, torch_input, use_unpack_operator, input_names, output_names, d_shapes = \
        get_block_and_inputs(predictor=predictor, block=block, img_shape=img_shape)

    print(" - Testing torch model ...", end="")
    test_torch_model(torch_model, torch_input, silent=True,
                     use_unpack_operator=use_unpack_operator)
    print("OK")

    print(" - Exporting to ONNX ...", end="")
    export_path = out_dir / Path(f"{m.name}-{block}-sa1-v01-op{opset_version}.onnx")

    with torch.no_grad():
        torch.onnx.export(
            torch_model,
            torch_input,
            str(export_path),
            export_params=True,
            opset_version=opset_version,
            input_names=input_names,
            output_names=output_names,
            dynamic_shapes=d_shapes,
            dynamo=True,
        )
    print("OK")
    print(f" - Model stored in {export_path}")

    if use_simplify:
        print(" - Simplifying model ...", end="")
        model = onnx.load(str(export_path))
        model, check = onnxsim.simplify(model)
        if check:
            export_path = modify_filename(export_path, str_extension="-simply")
            onnx.save(model, str(export_path))
            print("OK")
            print(f" - Simplified model stored in {export_path}")
        else:
            print("Model simplification failed!")

    print(" - Analysing model ...")
    analyze_onnx_model(onnx.load(str(export_path)))
    print(" - Model block successfully converted to ONNX.")

if __name__ == "__main__":
    import argparse
    print("---\n"
          "Welcome to Nanosam2 ONNX exporter!\n"
          "Nanosam2 ONNX exporter is used to export parts of the Nanosam2 model to ONNX files.\n"
          "These ONNX files can be executed using onnxruntime or further processed and deployed on the desired hardware.\n"
          "\n"
          " - Use parameter img_shape to set the input shape. Default is [3,512,512].\n"
          "---"
          )
    models = [
            ModelSource("sam2.1_small",        "results/sam2.1_hiera_small.pt",                             "../sam2_configs/sam2.1_hiera_s.yaml"),
            ModelSource("nanosam2-resnet18",    "results/sam2.1_hiera_s_resnet18/checkpoint.pth",             "../sam2_configs/nanosam2.1_resnet18.yaml"),
            ModelSource("nanosam2-mobilenetV3", "results/sam2.1_hiera_s_mobilenetV3_large/checkpoint.pth",    "../sam2_configs/nanosam2.1_mobilenet_v3_large.yaml"),
            ModelSource("nanosam2-casvit_s",    "results/sam2.1_hiera_s_casvit_s/checkpoint.pth",            "../sam2_configs/nanosam2.1_casvit_s.yaml"),
            ]

    valid_exports = [
        "nanosam2",
        "all",
        "image-encoder",         # trunk + neck + conv_s0/conv_s1
        "image-encoder-trunk",
        "image-encoder-neck",
        "sam-heads",             # prompt encoder + mask decoder (recommended)
        "mask-decoder",          # alias for sam-heads
        "mask-decoder-transformer",
        "memory-encoder",
        "memory-attention",
    ]

    valid_encoders = [
        "hiera_small",
        "resnet18",
        "mobilenetV3_large",
        "casvit_s",
    ]

    parser = argparse.ArgumentParser("Nanosam2 ONNX exporter")
    parser.add_argument("--export", type=str, default="image-encoder",
                        choices=valid_exports,
                        help="Model block to export.  Use 'all' to export all supported blocks.")
    parser.add_argument("--output_path", type=str, default="model_exports",
                        help="Export directory.")
    parser.add_argument("--img_shape", nargs="+", type=int, default=[3, 512, 512],
                        help="Image shape [C, H, W] to use.")
    parser.add_argument("--encoder_type", type=str, default="resnet18",
                        choices=valid_encoders)
    parser.add_argument("--opset", type=int, default=24,
                        help="ONNX opset version (default: 24)")
    parser.add_argument("--simplify", action="store_true", help="Simplify model with onnxsim")

    args = parser.parse_args()
    out_dir = Path(args.output_path)

    match args.encoder_type:
        case "hiera_small":
            model = models[0]
        case "resnet18":
            model = models[1]
        case "mobilenetV3_large":
            model = models[2]
        case "casvit_s":
            model = models[3]
        case _:
            print("Encoder type not supported.")
            exit(1)

    if not Path(model.checkpoint).is_file():
        print(f"Path to model checkpoint '{model.checkpoint}' not found.")
        exit(1)

    # Blocks that can be reliably exported to ONNX
    exportable_blocks = [
        "image-encoder",
        "image-encoder-trunk",
        "sam-heads",
        "mask-decoder",
        "memory-encoder",
        "memory-attention",
    ]

    if args.export == "all":
        for b in exportable_blocks:
            print(f"\nExporting block: {b}")
            export_model_block(model, b, out_dir, args.img_shape,
                               use_simplify=args.simplify, opset_version=args.opset)
    else:
        print(f"\nExporting block: {args.export}")
        export_model_block(model, args.export, out_dir, args.img_shape,
                           use_simplify=args.simplify, opset_version=args.opset)
    print("done.")
