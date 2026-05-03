# ONNX Camera Predictor for NanoSAM2
#
# A pure-ONNX replacement for SAM2CameraPredictor.  Once the ONNX files have
# been exported with tools/onnx_export.py, no PyTorch is required at inference
# time — only numpy and onnxruntime.
#
# Supported API (mirrors SAM2CameraPredictor):
#   predictor.load_first_frame(img)
#   predictor.add_new_points_or_box(frame_idx, obj_id, points, labels, box)
#   predictor.track(img)
#   predictor.reset_state()

from __future__ import annotations

import math
import warnings
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import onnxruntime as ort


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

IMG_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMG_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
NO_OBJ_SCORE = -1024.0


def _preprocess_image(img: np.ndarray, image_size: int = 512
                      ) -> Tuple[np.ndarray, int, int]:
    """Resize, normalise, and return (chw_float32, orig_w, orig_h)."""
    h, w = img.shape[:2]
    resized = cv2.resize(img, (image_size, image_size)).astype(np.float32) / 255.0
    resized = (resized - IMG_MEAN) / IMG_STD
    chw = resized.transpose(2, 0, 1)[np.newaxis]          # (1, 3, H, W)
    return chw.astype(np.float32), w, h


def _get_1d_sine_pe(pos: np.ndarray, dim: int, temperature: float = 10000.0
                    ) -> np.ndarray:
    """Numpy port of sam2_utils.get_1d_sine_pe."""
    pe_dim = dim // 2
    dim_t = np.arange(pe_dim, dtype=np.float32)
    dim_t = temperature ** (2 * (dim_t // 2) / pe_dim)
    pos_embed = pos[:, np.newaxis] / dim_t[np.newaxis, :]   # (N, pe_dim)
    return np.concatenate([np.sin(pos_embed), np.cos(pos_embed)], axis=-1)


# ---------------------------------------------------------------------------
# Session manager (thin wrapper around onnxruntime.InferenceSession)
# ---------------------------------------------------------------------------

class _OnnxSession:
    def __init__(self, path: str, providers=None):
        self._sess = ort.InferenceSession(
            path,
            providers=providers or ort.get_available_providers(),
        )
        self._in_names  = [inp.name for inp in self._sess.get_inputs()]
        self._out_names = [out.name for out in self._sess.get_outputs()]

    def run(self, **kwargs) -> List[np.ndarray]:
        """Run session.  Pass keyword arguments matching the ONNX input names."""
        feeds = {k: np.ascontiguousarray(v) for k, v in kwargs.items()
                 if k in self._in_names}
        return self._sess.run(None, feeds)


# ---------------------------------------------------------------------------
# NanoSAM2OnnxPredictor
# ---------------------------------------------------------------------------

class NanoSAM2OnnxPredictor:
    """
    Pure ONNX / numpy camera predictor for NanoSAM2.

    Parameters
    ----------
    model_dir : str or Path
        Directory that contains the 4 ONNX files produced by onnx_export.py:
            *-image-encoder-*.onnx
            *-sam-heads-*.onnx  (or *-mask-decoder-*.onnx)
            *-memory-encoder-*.onnx
            *-memory-attention-*.onnx
    encoder_prefix : str
        File name prefix, e.g. "nanosam2-resnet18".  If None, the first match
        in model_dir is used.
    image_size : int
        Must match the image size the model was exported with (default 512).
    num_maskmem : int
        Size of the rolling memory bank (default 7, same as the pytorch model).
    sigmoid_scale : float
        Scale factor applied before feeding masks to memory encoder (20.0).
    sigmoid_bias : float
        Bias applied before feeding masks to memory encoder (-10.0).
    providers : list[str] | None
        OnnxRuntime execution providers, e.g. ["CUDAExecutionProvider",
        "CPUExecutionProvider"].  None → auto-detect.
    """

    def __init__(
        self,
        model_dir: Union[str, Path],
        encoder_prefix: Optional[str] = None,
        image_size: int = 512,
        num_maskmem: int = 7,
        max_obj_ptrs_in_encoder: int = 16,
        sigmoid_scale: float = 20.0,
        sigmoid_bias: float = -10.0,
        hidden_dim: int = 256,
        mem_dim: int = 64,
        directly_add_no_mem_embed: bool = True,
        use_obj_ptrs_in_encoder: bool = True,
        add_tpos_enc_to_obj_ptrs: bool = True,
        proj_tpos_enc_in_obj_ptrs: bool = True,
        use_signed_tpos_enc_to_obj_ptrs: bool = True,
        only_obj_ptrs_in_the_past: bool = True,
        providers: Optional[List[str]] = None,
        non_overlap_masks: bool = False,
    ):
        model_dir = Path(model_dir)
        self.image_size   = image_size
        self.num_maskmem  = num_maskmem
        self.max_obj_ptrs = max_obj_ptrs_in_encoder
        self.sigmoid_scale = sigmoid_scale
        self.sigmoid_bias  = sigmoid_bias
        self.hidden_dim    = hidden_dim
        self.mem_dim       = mem_dim
        self.directly_add_no_mem_embed       = directly_add_no_mem_embed
        self.use_obj_ptrs_in_encoder         = use_obj_ptrs_in_encoder
        self.add_tpos_enc_to_obj_ptrs        = add_tpos_enc_to_obj_ptrs
        self.proj_tpos_enc_in_obj_ptrs       = proj_tpos_enc_in_obj_ptrs
        self.use_signed_tpos_enc_to_obj_ptrs = use_signed_tpos_enc_to_obj_ptrs
        self.only_obj_ptrs_in_the_past       = only_obj_ptrs_in_the_past
        self.non_overlap_masks               = non_overlap_masks

        # Spatial resolution at stride-16 level
        self.h16 = image_size // 16   # 32
        self.w16 = image_size // 16   # 32
        self.hw16 = self.h16 * self.w16  # 1024

        # ---- locate ONNX files ------------------------------------------
        def _find(pattern: str) -> str:
            matches = sorted(model_dir.glob(f"*{pattern}*.onnx"))
            if encoder_prefix:
                matches = [m for m in matches if m.name.startswith(encoder_prefix)]
            # Prefer the model with the highest opset (last alphabetically when
            # file names include "-opXX.onnx")
            matches = sorted(matches, key=lambda p: p.name)
            if not matches:
                raise FileNotFoundError(
                    f"No ONNX file matching '*{pattern}*.onnx' in {model_dir}"
                )
            return str(matches[-1])  # highest opset / latest export

        self._img_enc_sess    = _OnnxSession(_find("image-encoder-sa1"), providers)
        # sam-heads preferred over legacy mask-decoder
        try:
            self._sam_heads_sess = _OnnxSession(_find("sam-heads"),    providers)
        except FileNotFoundError:
            self._sam_heads_sess = _OnnxSession(_find("mask-decoder"), providers)
        self._mem_enc_sess    = _OnnxSession(_find("memory-encoder"),  providers)
        self._mem_attn_sess   = _OnnxSession(_find("memory-attention"),providers)

        # ---- model constants (loaded from a companion .npz file if present,
        #      otherwise the user must call load_constants_from_checkpoint) ----
        npz_candidates = sorted(model_dir.glob("*constants*.npz"))
        if encoder_prefix:
            npz_candidates = [p for p in npz_candidates
                              if p.name.startswith(encoder_prefix)]
        if npz_candidates:
            self._load_constants_npz(str(npz_candidates[0]))
        else:
            raise FileNotFoundError(
                f"No constants .npz file found in {model_dir} "
                f"(prefix: {encoder_prefix!r}). "
                "Run 'python tools/onnx_export.py --export all' to generate it."
            )

        # ---- inference state -----------------------------------------------
        self.condition_state: Dict = {}
        self.frame_idx = 0

    # ------------------------------------------------------------------
    # Constants
    # ------------------------------------------------------------------

    def _load_constants_npz(self, path: str) -> None:
        data = np.load(path, allow_pickle=False)
        self.no_mem_embed    = data["no_mem_embed"]
        self.no_mem_pos_enc  = data["no_mem_pos_enc"]
        self.maskmem_tpos_enc = data["maskmem_tpos_enc"]
        w = data["tpos_proj_w"]
        b = data["tpos_proj_b"]
        self._tpos_proj_w = w if w.size > 0 else None
        self._tpos_proj_b = b if b.size > 0 else None
        self._constants_loaded = True

    def _check_constants(self):
        if not self._constants_loaded:
            raise RuntimeError(
                "Model constants not loaded.  Call "
                "load_constants_from_checkpoint() or provide a constants .npz file."
            )

    # ------------------------------------------------------------------
    # Image encoding helpers
    # ------------------------------------------------------------------

    def _run_image_encoder(self, chw: np.ndarray):
        """chw: (1, 3, H, W) float32  →  dict of named feature tensors."""
        outputs = self._img_enc_sess.run(image=chw)
        names   = self._img_enc_sess._out_names
        return dict(zip(names, outputs))

    def _get_feature_maps(self, chw: np.ndarray):
        """
        Run image encoder and return the quantities needed by downstream modules.
        Returns
        -------
        hr0 : (1, C/8, H/4,  W/4)  high-res projected feature
        hr1 : (1, C/4, H/8,  W/8)
        img_emb : (1, C, H/16, W/16)   = image_embedding
        pos_enc : (1, C, H/16, W/16)   = vision_pos_enc
        """
        outputs = self._img_enc_sess.run(image=chw)
        # Use output index order (matches the export order):
        # 0: high_res_feat_0, 1: high_res_feat_1, 2: image_embedding,
        # 3: vision_pos_enc, 4: backbone_fpn_raw_0, 5: backbone_fpn_raw_1
        hr0    = outputs[0]
        hr1    = outputs[1]
        img_emb = outputs[2]
        pos_enc = outputs[3]
        return hr0, hr1, img_emb, pos_enc

    # ------------------------------------------------------------------
    # Memory bank helpers (numpy)
    # ------------------------------------------------------------------

    def _build_memory_tensors(self, frame_idx: int, num_frames: int
                               ) -> Tuple[np.ndarray, np.ndarray, int]:
        """
        Build (memory, memory_pos, num_obj_ptr_tokens) arrays for memory_attention.
        Mirrors SAM2Base._prepare_memory_conditioned_features (non-init branch).
        """
        output_dict = self.condition_state["output_dict"]
        cond_outs   = output_dict["cond_frame_outputs"]
        non_cond_outs = output_dict["non_cond_frame_outputs"]

        # cached spatial pos enc (same for all frames)
        maskmem_pos_enc_const = self.condition_state["constants"].get(
            "maskmem_pos_enc", None
        )  # (1, mem_dim, h16, w16)

        to_cat_memory     = []
        to_cat_memory_pos = []

        # ---- conditioning frames (t_pos=0) -----------------------------------
        # Use the single init conditioning frame
        for t_idx, out in cond_outs.items():
            feats = out["maskmem_features"]  # (1, mem_dim, h16, w16)
            flat  = feats.reshape(self.mem_dim, -1).T[:, np.newaxis, :]  # (hw16, 1, md)
            to_cat_memory.append(flat)

            enc = maskmem_pos_enc_const  # (1, mem_dim, h16, w16)
            enc_flat = enc.reshape(self.mem_dim, -1).T[:, np.newaxis, :]  # (hw16,1,md)
            # t_pos=0 → maskmem_tpos_enc[num_maskmem - 0 - 1] = tpos_enc[6]
            tpos_idx = self.num_maskmem - 0 - 1
            tpos = self.maskmem_tpos_enc[tpos_idx]  # (1, 1, md)
            enc_flat = enc_flat + tpos
            to_cat_memory_pos.append(enc_flat)

        # ---- non-conditioning frames (t_pos=1..num_maskmem-1) ----------------
        stride = 1  # camera predictor always uses stride 1
        for t_pos in range(1, self.num_maskmem):
            t_rel = self.num_maskmem - t_pos
            if t_rel == 1:
                prev_idx = frame_idx - 1
            else:
                prev_idx = ((frame_idx - 2) // stride) * stride
                prev_idx = prev_idx - (t_rel - 2) * stride
            out = non_cond_outs.get(prev_idx, None)
            if out is None:
                continue
            feats = out["maskmem_features"]
            flat  = feats.reshape(self.mem_dim, -1).T[:, np.newaxis, :]
            to_cat_memory.append(flat)

            enc = maskmem_pos_enc_const
            enc_flat = enc.reshape(self.mem_dim, -1).T[:, np.newaxis, :]
            tpos_idx = self.num_maskmem - t_pos - 1
            tpos = self.maskmem_tpos_enc[tpos_idx]
            enc_flat = enc_flat + tpos
            to_cat_memory_pos.append(enc_flat)

        # ---- object pointer tokens -------------------------------------------
        num_obj_ptr_tokens = 0
        if self.use_obj_ptrs_in_encoder:
            max_ptrs = min(num_frames, self.max_obj_ptrs)
            tpos_sign = 1  # forward tracking

            pos_and_ptrs = []
            # From conditioning frames
            for t_idx, out in cond_outs.items():
                if "obj_ptr" not in out:
                    continue
                if (self.only_obj_ptrs_in_the_past and
                        t_idx > frame_idx):
                    continue
                t_diff = (frame_idx - t_idx) * tpos_sign if self.use_signed_tpos_enc_to_obj_ptrs \
                         else abs(frame_idx - t_idx)
                pos_and_ptrs.append((t_diff, out["obj_ptr"]))  # obj_ptr: (1, C)

            # From non-conditioning frames
            for t_diff_val in range(1, max_ptrs):
                t = frame_idx - t_diff_val
                if t < 0:
                    break
                out = non_cond_outs.get(t, None)
                if out is None or "obj_ptr" not in out:
                    continue
                d = t_diff_val * tpos_sign if self.use_signed_tpos_enc_to_obj_ptrs \
                    else t_diff_val
                pos_and_ptrs.append((d, out["obj_ptr"]))

            if pos_and_ptrs:
                t_diff_max = max_ptrs - 1
                obj_ptrs_list = []
                obj_pos_list  = []
                for d, ptr in pos_and_ptrs:
                    # ptr: (1, C=256)
                    obj_ptrs_list.append(ptr[np.newaxis])  # (1, 1, 256)
                    if self.add_tpos_enc_to_obj_ptrs:
                        tpos_val = np.array([d], dtype=np.float32) / max(t_diff_max, 1)
                        # sine PE: (1, C)
                        tpos_dim = self.hidden_dim  # 256 (projected later)
                        tpos_pe  = _get_1d_sine_pe(tpos_val, tpos_dim)  # (1, 256)
                        if self._tpos_proj_w is not None:
                            # Linear projection: (1, mem_dim)
                            tpos_pe = tpos_pe @ self._tpos_proj_w.T + self._tpos_proj_b
                        # Expand to (1, 1, mem_dim)
                        tpos_pe = tpos_pe[:, np.newaxis, :]  # (1, 1, md)
                    else:
                        tpos_pe = np.zeros((1, 1, self.mem_dim), dtype=np.float32)
                    obj_pos_list.append(tpos_pe)

                # Stack: (n_ptrs, 1, C=256)
                obj_ptrs = np.concatenate(obj_ptrs_list, axis=0)

                # Split each pointer into (C // mem_dim) tokens of mem_dim
                C, md = self.hidden_dim, self.mem_dim
                if md < C:
                    ratio = C // md
                    # (n_ptrs, 1, C) → (n_ptrs, 1, ratio, md) → (n_ptrs*ratio, 1, md)
                    obj_ptrs = obj_ptrs.reshape(len(pos_and_ptrs), 1, ratio, md)
                    obj_ptrs = obj_ptrs.transpose(0, 2, 1, 3).reshape(-1, 1, md)
                    # Repeat pos accordingly
                    obj_pos_arr = np.concatenate(obj_pos_list, axis=0)  # (n_ptrs,1,md)
                    obj_pos_arr = np.repeat(obj_pos_arr, ratio, axis=0)
                else:
                    obj_ptrs = obj_ptrs.reshape(-1, 1, md)
                    obj_pos_arr = np.concatenate(obj_pos_list, axis=0)

                to_cat_memory.append(obj_ptrs)
                to_cat_memory_pos.append(obj_pos_arr)
                num_obj_ptr_tokens = obj_ptrs.shape[0]

        if not to_cat_memory:
            # No memory frames available yet (can happen on the very first track
            # call when conditioning-frame storage is empty).  Signal the caller
            # to apply no_mem_embed directly instead of running attention.
            return None, None, 0

        memory     = np.concatenate(to_cat_memory,     axis=0).astype(np.float32)
        memory_pos = np.concatenate(to_cat_memory_pos, axis=0).astype(np.float32)
        return memory, memory_pos, num_obj_ptr_tokens

    def _run_memory_attention(self, img_emb: np.ndarray, pos_enc: np.ndarray,
                               frame_idx: int, num_frames: int) -> np.ndarray:
        """
        Run memory attention and return the conditioned image embedding
        (1, C, h16, w16).
        """
        # Flatten img_emb + pos_enc to seq-first: (hw16, 1, C)
        curr     = img_emb.reshape(self.hidden_dim, -1).T[:, np.newaxis, :]
        curr_pos = pos_enc.reshape(self.hidden_dim, -1).T[:, np.newaxis, :]

        memory, memory_pos, _ = self._build_memory_tensors(frame_idx, num_frames)

        if memory is None:
            # No memory available yet — mirror the PyTorch directly_add_no_mem_embed
            # path: add the learned no-memory embedding to the current features and
            # return without running the (expensive) memory-attention ONNX session.
            conditioned = curr + self.no_mem_embed  # (hw16,1,C) + (1,1,C) broadcast
            return conditioned[:, 0, :].T.reshape(
                1, self.hidden_dim, self.h16, self.w16
            )

        attended = self._mem_attn_sess.run(
            curr=curr.astype(np.float32),
            curr_pos=curr_pos.astype(np.float32),
            memory=memory,
            memory_pos=memory_pos,
        )[0]  # (hw16, 1, C)

        # Reshape back to (1, C, h16, w16)
        return attended[:, 0, :].T.reshape(1, self.hidden_dim, self.h16, self.w16)

    def _run_sam_heads(self, img_emb_cond: np.ndarray, hr0: np.ndarray,
                       hr1: np.ndarray, point_coords: np.ndarray,
                       point_labels: np.ndarray):
        """Run SAM heads and return (low_res_masks, high_res_masks, obj_ptr, obj_score)."""
        outputs = self._sam_heads_sess.run(
            image_embedding=img_emb_cond.astype(np.float32),
            high_res_feat_0=hr0.astype(np.float32),
            high_res_feat_1=hr1.astype(np.float32),
            point_coords=point_coords.astype(np.float32),
            point_labels=point_labels.astype(np.int32),
        )
        low_res_masks, high_res_masks, iou_preds, obj_ptr, obj_score_logits = outputs
        return low_res_masks, high_res_masks, obj_ptr, obj_score_logits

    def _run_memory_encoder(self, img_emb_raw: np.ndarray,
                             high_res_masks: np.ndarray,
                             obj_score_logits: np.ndarray):
        """
        Run memory encoder and return (maskmem_features, maskmem_pos_enc).
        img_emb_raw: (1, C, h16, w16)  – RAW backbone feature (NOT attended)
        """
        # scale mask: sigmoid(high_res_masks) * scale + bias
        mask_for_mem = 1.0 / (1.0 + np.exp(-high_res_masks))  # sigmoid
        mask_for_mem = mask_for_mem * self.sigmoid_scale + self.sigmoid_bias

        outputs = self._mem_enc_sess.run(
            pix_feat=img_emb_raw.astype(np.float32),
            mask=mask_for_mem.astype(np.float32),
        )
        maskmem_features, maskmem_pos_enc = outputs[:2]
        return maskmem_features, maskmem_pos_enc

    # ------------------------------------------------------------------
    # Object management
    # ------------------------------------------------------------------

    def _init_state(self):
        self.condition_state = {
            "num_frames": 1,
            "video_height": None,
            "video_width":  None,
            "obj_id_to_idx": OrderedDict(),
            "obj_idx_to_id": OrderedDict(),
            "obj_ids": [],
            "point_inputs_per_obj": {},
            "output_dict": {
                "cond_frame_outputs":     {},
                "non_cond_frame_outputs": {},
            },
            "output_dict_per_obj": {},
            "temp_output_dict_per_obj": {},
            "consolidated_frame_inds": {
                "cond_frame_outputs":     set(),
                "non_cond_frame_outputs": set(),
            },
            "tracking_has_started": False,
            "frames_already_tracked": {},
            "constants": {},
            # cache for image features of the first frame
            "first_frame_feats": None,
        }

    def _obj_id_to_idx(self, obj_id: int) -> int:
        if obj_id in self.condition_state["obj_id_to_idx"]:
            return self.condition_state["obj_id_to_idx"][obj_id]
        if self.condition_state["tracking_has_started"]:
            raise RuntimeError(
                f"Cannot add object {obj_id} after tracking has started. "
                "Call reset_state() first."
            )
        idx = len(self.condition_state["obj_id_to_idx"])
        self.condition_state["obj_id_to_idx"][obj_id] = idx
        self.condition_state["obj_idx_to_id"][idx]    = obj_id
        self.condition_state["obj_ids"] = list(self.condition_state["obj_id_to_idx"])
        self.condition_state["point_inputs_per_obj"][idx] = {}
        self.condition_state["output_dict_per_obj"][idx]  = {
            "cond_frame_outputs": {}, "non_cond_frame_outputs": {}
        }
        self.condition_state["temp_output_dict_per_obj"][idx] = {
            "cond_frame_outputs": {}, "non_cond_frame_outputs": {}
        }
        return idx

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_first_frame(self, img: np.ndarray) -> None:
        """
        Preprocess and encode the first frame.  Call add_new_points_or_box()
        afterwards to set the initial prompt.
        """
        self._check_constants()
        self._init_state()
        self.frame_idx = 0

        chw, w, h = _preprocess_image(img, self.image_size)
        self.condition_state["video_width"]  = w
        self.condition_state["video_height"] = h
        self.condition_state["num_frames"]   = 1

        hr0, hr1, img_emb, pos_enc = self._get_feature_maps(chw)
        self.condition_state["first_frame_feats"] = (hr0, hr1, img_emb, pos_enc)

    def add_new_points_or_box(
        self,
        frame_idx: int,
        obj_id:    int,
        points:    Optional[np.ndarray] = None,
        labels:    Optional[np.ndarray] = None,
        box:       Optional[np.ndarray] = None,
        normalize_coords: bool = True,
    ) -> Tuple[int, List[int], np.ndarray]:
        """
        Add a click or box prompt on frame 0 (before tracking starts).

        Returns (frame_idx, obj_ids, video_res_masks).
        """
        self._check_constants()
        obj_idx = self._obj_id_to_idx(obj_id)

        # ---- build point_coords / point_labels numpy arrays ----------------
        if points is None:
            pts = np.zeros((1, 0, 2), dtype=np.float32)
        else:
            pts = np.asarray(points, dtype=np.float32)
            if pts.ndim == 2:
                pts = pts[np.newaxis]  # (1, N, 2)

        if labels is None:
            lbs = np.zeros((1, 0), dtype=np.int32)
        else:
            lbs = np.asarray(labels, dtype=np.int32)
            if lbs.ndim == 1:
                lbs = lbs[np.newaxis]  # (1, N)

        if box is not None:
            box = np.asarray(box, dtype=np.float32).reshape(1, 2, 2)
            box_labels = np.array([[2, 3]], dtype=np.int32)
            pts = np.concatenate([box, pts], axis=1)
            lbs = np.concatenate([box_labels, lbs], axis=1)

        if normalize_coords:
            W = self.condition_state["video_width"]
            H = self.condition_state["video_height"]
            pts = pts / np.array([[[W, H]]], dtype=np.float32)

        pts = pts * self.image_size   # scale to model pixel space

        # ---- run SAM heads on first frame -----------------------------------
        hr0, hr1, img_emb, pos_enc = self.condition_state["first_frame_feats"]

        # On the initial conditioning frame with directly_add_no_mem_embed,
        # we add the no-memory embedding instead of running memory attention.
        if self.directly_add_no_mem_embed:
            # curr_feats: (hw16, 1, C) + no_mem_embed (1, 1, C)
            curr_flat = img_emb.reshape(self.hidden_dim, -1).T[:, np.newaxis, :]
            cond_flat = curr_flat + self.no_mem_embed  # broadcast
            img_emb_cond = cond_flat[:, 0, :].T.reshape(
                1, self.hidden_dim, self.h16, self.w16
            )
        else:
            img_emb_cond = img_emb.copy()

        low_res, high_res, obj_ptr, obj_score = self._run_sam_heads(
            img_emb_cond, hr0, hr1, pts, lbs
        )

        # ---- store in temp_output_dict_per_obj (no memory encoding yet) -----
        out = {
            "pred_masks":      low_res,
            "pred_masks_high_res": high_res,
            "obj_ptr":         obj_ptr,
            "object_score_logits": obj_score,
            "maskmem_features": None,
            "maskmem_pos_enc":  None,
        }
        self.condition_state["temp_output_dict_per_obj"][obj_idx][
            "cond_frame_outputs"
        ][frame_idx] = out

        # ---- build video-resolution masks ----------------------------------
        obj_ids  = self.condition_state["obj_ids"]
        vid_masks = self._upscale_to_video_res(high_res)
        return frame_idx, obj_ids, vid_masks

    def _propagate_in_video_preflight(self) -> None:
        """Encode conditioning frames into memory (run memory encoder)."""
        self.condition_state["tracking_has_started"] = True
        output_dict = self.condition_state["output_dict"]
        temp_dicts  = self.condition_state["temp_output_dict_per_obj"]

        hr0, hr1, img_emb_raw, pos_enc = self.condition_state["first_frame_feats"]

        for obj_idx, temp in temp_dicts.items():
            for frame_idx, out in temp["cond_frame_outputs"].items():
                # Run memory encoder to create maskmem_features
                high_res = out["pred_masks_high_res"]
                obj_score = out["object_score_logits"]
                maskmem_feat, maskmem_pos = self._run_memory_encoder(
                    img_emb_raw, high_res, obj_score
                )

                # Cache maskmem_pos_enc as a constant (same for all frames)
                if "maskmem_pos_enc" not in self.condition_state["constants"]:
                    self.condition_state["constants"]["maskmem_pos_enc"] = maskmem_pos

                out["maskmem_features"] = maskmem_feat
                out["maskmem_pos_enc"]  = [maskmem_pos]

                # Store in main output_dict
                if frame_idx not in output_dict["cond_frame_outputs"]:
                    output_dict["cond_frame_outputs"][frame_idx] = {}
                output_dict["cond_frame_outputs"][frame_idx] = out
                self.condition_state["consolidated_frame_inds"][
                    "cond_frame_outputs"
                ].add(frame_idx)

            temp["cond_frame_outputs"].clear()

    def track(
        self,
        img: np.ndarray,
        frame_idx: Optional[int] = None,
    ) -> Tuple[List[int], np.ndarray]:
        """
        Track objects in the next camera frame.

        Parameters
        ----------
        img : np.ndarray  RGB image (H, W, 3) uint8.
        frame_idx : int | None  Override frame counter.

        Returns (obj_ids, video_res_masks).
        """
        self._check_constants()

        if frame_idx is None:
            self.frame_idx += 1
            self.condition_state["num_frames"] += 1
        else:
            self.frame_idx = frame_idx
            if frame_idx >= self.condition_state["num_frames"]:
                self.condition_state["num_frames"] = frame_idx + 1

        if not self.condition_state["tracking_has_started"]:
            self._propagate_in_video_preflight()

        chw, _, _ = _preprocess_image(img, self.image_size)
        hr0, hr1, img_emb, pos_enc = self._get_feature_maps(chw)

        # ---- memory-conditioned features -----------------------------------
        img_emb_cond = self._run_memory_attention(
            img_emb, pos_enc, self.frame_idx,
            self.condition_state["num_frames"]
        )

        # ---- SAM heads (no prompts) ----------------------------------------
        # Empty padding point with label -1
        pts = np.zeros((1, 1, 2), dtype=np.float32)
        lbs = np.full((1, 1), -1, dtype=np.int32)

        low_res, high_res, obj_ptr, obj_score = self._run_sam_heads(
            img_emb_cond, hr0, hr1, pts, lbs
        )

        # ---- memory encoder -----------------------------------------------
        maskmem_feat, maskmem_pos = self._run_memory_encoder(
            img_emb, high_res, obj_score   # img_emb: raw backbone feat
        )

        # Cache maskmem_pos_enc if not already done
        if "maskmem_pos_enc" not in self.condition_state["constants"]:
            self.condition_state["constants"]["maskmem_pos_enc"] = maskmem_pos

        # ---- store in non-conditioning memory bank -------------------------
        current_out = {
            "maskmem_features": maskmem_feat,
            "maskmem_pos_enc":  [maskmem_pos],
            "pred_masks":       low_res,
            "obj_ptr":          obj_ptr,
        }
        self._manage_memory(self.frame_idx, current_out)

        obj_ids   = self.condition_state["obj_ids"]
        vid_masks = self._upscale_to_video_res(high_res)
        return obj_ids, vid_masks

    def _manage_memory(self, frame_idx: int, current_out: dict) -> None:
        """Rolling memory bank: keep at most num_maskmem frames."""
        nc = self.condition_state["output_dict"]["non_cond_frame_outputs"]
        nc[frame_idx] = current_out
        keys = list(nc.keys())
        if len(nc) > self.num_maskmem:
            for k in keys[:len(nc) - self.num_maskmem]:
                del nc[k]

    def _upscale_to_video_res(self, high_res_masks: np.ndarray) -> np.ndarray:
        """Resize (B, 1, H_model, W_model) to (B, 1, video_H, video_W)."""
        vH = self.condition_state["video_height"]
        vW = self.condition_state["video_width"]
        if high_res_masks.shape[-2:] == (vH, vW):
            return high_res_masks
        B, C, H, W = high_res_masks.shape
        out = np.stack([
            cv2.resize(high_res_masks[b, c], (vW, vH), interpolation=cv2.INTER_LINEAR)
            for b in range(B) for c in range(C)
        ]).reshape(B, C, vH, vW)
        if self.non_overlap_masks and B > 1:
            best = out.argmax(axis=0, keepdims=True)   # (1,1,H,W)
            idx  = np.arange(B)[:, np.newaxis, np.newaxis, np.newaxis]
            keep = best == idx
            out  = np.where(keep, out, np.minimum(out, -10.0))
        return out

    def reset_state(self) -> None:
        """Clear all tracking state."""
        self._init_state()
        self.frame_idx = 0


# ---------------------------------------------------------------------------
# Factory function (mirrors build_sam2_camera_predictor API)
# ---------------------------------------------------------------------------

def build_onnx_camera_predictor(
    model_dir: Union[str, Path],
    encoder_prefix: Optional[str] = None,
    image_size: int = 512,
    providers: Optional[List[str]] = None,
    # NanoSAM2 resnet18 defaults (override if using a different model):
    num_maskmem: int = 7,
    sigmoid_scale: float = 20.0,
    sigmoid_bias:  float = -10.0,
    hidden_dim: int = 256,
    mem_dim:    int = 64,
) -> NanoSAM2OnnxPredictor:
    """
    Build a NanoSAM2OnnxPredictor.

    The constants .npz file (created by ``tools/onnx_export.py``) must be
    present in *model_dir* alongside the ONNX files.  No PyTorch is required.

    Parameters
    ----------
    model_dir      : directory containing the ONNX files and the constants .npz.
    encoder_prefix : e.g. "nanosam2-resnet18" (optional — auto-detected if None).
    """
    return NanoSAM2OnnxPredictor(
        model_dir=model_dir,
        encoder_prefix=encoder_prefix,
        image_size=image_size,
        num_maskmem=num_maskmem,
        sigmoid_scale=sigmoid_scale,
        sigmoid_bias=sigmoid_bias,
        hidden_dim=hidden_dim,
        mem_dim=mem_dim,
        providers=providers,
    )
