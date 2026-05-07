# NanoSAM2 Model Comparison Demo
#
# Run the PyTorch and ONNX models side-by-side on the same video source and
# compare their outputs.  Produces:
#   - A live window showing both masks next to each other.
#   - An optional directory of saved mask images (--save_masks).
#   - A CSV file with per-frame MSE and IoU metrics (--csv).
#
# Usage:
#   python demos/compare_models.py \
#       --config nanosam2.1_resnet18.yaml \
#       --checkpoint results/sam2.1_hiera_s_resnet18/checkpoint.pth \
#       --model_dir model_exports \
#       --video videos/bedroom \
#       --box 230,134,294,219 \
#       [--box 353,11,451,122] \
#       [--save_masks mask_output] \
#       [--csv comparison.csv] \
#       [--max_frames 200]

import argparse
import csv
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from demos.demo_utils import (  # noqa: E402
    draw_masks, open_video_source, parse_boxes, read_next_frame,
    torch_masks_to_numpy,
)
from nanosam2.sam2.build_sam import build_sam2_camera_predictor  # noqa: E402
from nanosam2.sam2.sam2_onnx_camera_predictor import (  # noqa: E402
    build_onnx_camera_predictor,
)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x.clip(-50, 50)))


def _logits_to_binary(logits: np.ndarray) -> np.ndarray:
    """Convert (N, 1, H, W) logits to a combined binary mask (H, W)."""
    probs = _sigmoid(logits)
    binary = (probs > 0.5).astype(np.float32)
    # Merge all objects into one mask for comparison
    return binary.max(axis=0).squeeze(0)  # (H, W)


def compute_mse(ref: np.ndarray, test: np.ndarray) -> float:
    """Mean squared error between two (H, W) float masks."""
    return float(np.mean((ref - test) ** 2))


def compute_iou(ref: np.ndarray, test: np.ndarray) -> float:
    """Intersection over union between two binary (H, W) masks."""
    intersection = np.logical_and(ref > 0.5, test > 0.5).sum()
    union = np.logical_or(ref > 0.5, test > 0.5).sum()
    if union == 0:
        return 1.0  # both empty
    return float(intersection / union)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(
        description="Compare NanoSAM2 PyTorch vs ONNX model outputs"
    )
    # PyTorch model
    p.add_argument("--config", default="nanosam2.1_resnet18",
                   help="SAM2 config name (default: nanosam2.1_resnet18)")
    p.add_argument("--checkpoint",
                   default="results/sam2.1_hiera_s_resnet18/checkpoint.pth",
                   help="PyTorch checkpoint path")
    p.add_argument("--device", default="cpu",
                   help="PyTorch device (default: cpu)")
    # ONNX model
    p.add_argument("--model_dir", default="model_exports",
                   help="Directory with ONNX files")
    p.add_argument("--encoder_prefix", default="nanosam2-resnet18",
                   help="ONNX file prefix")
    p.add_argument("--image_size", type=int, default=512,
                   help="Model input resolution (default: 512)")
    # Video source & prompts
    p.add_argument("--video", required=True,
                   help="Path to video file or image directory")
    p.add_argument("--box", dest="boxes", action="append", required=True,
                   metavar="x1,y1,x2,y2",
                   help="Bounding box prompt (repeatable)")
    # Output
    p.add_argument("--save_masks", default=None, metavar="DIR",
                   help="Directory to write per-frame mask images")
    p.add_argument("--csv", default="comparison.csv", metavar="FILE",
                   help="Path for the per-frame metrics CSV (default: comparison.csv)")
    p.add_argument("--max_frames", type=int, default=0,
                   help="Stop after this many frames (0 = unlimited)")
    p.add_argument("--no_display", action="store_true",
                   help="Disable the live window (headless mode)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = _parse_args()

    # ---- parse boxes -------------------------------------------------------
    try:
        boxes = parse_boxes(args.boxes)
    except ValueError as e:
        print(f"[ERROR] {e}")
        sys.exit(1)

    # The ONNX predictor currently tracks a single object at a time.
    # Use only the first box for a fair comparison.
    if len(boxes) > 1:
        print(f"[WARNING] The ONNX predictor supports single-object tracking. "
              f"Using only the first box; ignoring {len(boxes) - 1} extra box(es).")
    box = boxes[0]

    # ---- set up mask output dir --------------------------------------------
    mask_dir: Path | None = None
    if args.save_masks:
        mask_dir = Path(args.save_masks)
        mask_dir.mkdir(parents=True, exist_ok=True)

    # ---- build both predictors ---------------------------------------------
    print("Building PyTorch predictor ...")
    device = args.device
    if device == "cuda":
        torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
    pt_predictor = build_sam2_camera_predictor(
        args.config, args.checkpoint, device=torch.device(device)
    )

    print("Building ONNX predictor ...")
    ox_predictor = build_onnx_camera_predictor(
        model_dir=args.model_dir,
        encoder_prefix=args.encoder_prefix,
        image_size=args.image_size,
    )
    print("Both predictors ready.\n")

    # ---- open video source -------------------------------------------------
    cap, image_dir_iter, src = open_video_source(args.video)

    # ---- CSV setup ---------------------------------------------------------
    csv_path = Path(args.csv)
    csv_file = open(csv_path, "w", newline="")
    writer = csv.writer(csv_file)
    writer.writerow(["frame", "mse", "iou", "pt_time_ms", "ox_time_ms"])

    # ---- display -----------------------------------------------------------
    win = "PyTorch vs ONNX"
    if not args.no_display:
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win, 1600, 450)

    frame_idx = 0
    mse_accum = []

    while True:
        bgr = read_next_frame(cap, image_dir_iter, src)
        if bgr is None:
            break

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        # ---- first frame: init both predictors with same boxes -------------
        if frame_idx == 0:
            pt_predictor.load_first_frame(rgb)
            ox_predictor.load_first_frame(rgb)

            x1, y1, x2, y2 = box
            bbox_np = np.array([[x1, y1], [x2, y2]], dtype=np.float32)

            _, pt_obj_ids, pt_ml = pt_predictor.add_new_prompt(
                frame_idx=0, obj_id=1, bbox=bbox_np,
            )
            _, ox_obj_ids, ox_ml = ox_predictor.add_new_points_or_box(
                frame_idx=0, obj_id=1, box=bbox_np,
                normalize_coords=False,
            )

            pt_masks_np = torch_masks_to_numpy(pt_ml)
            ox_masks_np = ox_ml  # already numpy (1, 1, H, W)

        else:
            # ---- track on subsequent frames --------------------------------
            t0 = time.perf_counter()
            pt_obj_ids, pt_mask_raw = pt_predictor.track(rgb)
            pt_time = time.perf_counter() - t0

            t0 = time.perf_counter()
            ox_obj_ids, ox_mask_raw = ox_predictor.track(rgb)
            ox_time = time.perf_counter() - t0

            pt_masks_np = torch_masks_to_numpy(pt_mask_raw)
            ox_masks_np = ox_mask_raw

            # ---- metrics ---------------------------------------------------
            pt_bin = _logits_to_binary(pt_masks_np)
            ox_bin = _logits_to_binary(ox_masks_np)
            mse = compute_mse(pt_bin, ox_bin)
            iou = compute_iou(pt_bin, ox_bin)
            mse_accum.append(mse)

            writer.writerow([
                frame_idx,
                f"{mse:.6f}",
                f"{iou:.4f}",
                f"{pt_time * 1000:.1f}",
                f"{ox_time * 1000:.1f}",
            ])

        # ---- draw side by side ---------------------------------------------
        disp_pt = draw_masks(bgr.copy(), pt_obj_ids, pt_masks_np)
        disp_ox = draw_masks(bgr.copy(), ox_obj_ids, ox_masks_np)

        # Labels
        cv2.putText(disp_pt, "PyTorch", (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(disp_ox, "ONNX", (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)

        combined = np.hstack([disp_pt, disp_ox])

        # Frame counter
        cv2.putText(combined, f"Frame {frame_idx}", (10, combined.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)

        if not args.no_display:
            cv2.imshow(win, combined)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break

        # ---- save masks ----------------------------------------------------
        if mask_dir is not None:
            pt_bin_u8 = (_logits_to_binary(pt_masks_np) * 255).astype(np.uint8)
            ox_bin_u8 = (_logits_to_binary(ox_masks_np) * 255).astype(np.uint8)
            cv2.imwrite(str(mask_dir / f"pt_mask_{frame_idx:05d}.png"), pt_bin_u8)
            cv2.imwrite(str(mask_dir / f"ox_mask_{frame_idx:05d}.png"), ox_bin_u8)

        frame_idx += 1
        if args.max_frames > 0 and frame_idx >= args.max_frames:
            break

        if frame_idx % 50 == 0:
            print(f"  Processed {frame_idx} frames ...")

    # ---- clean up ----------------------------------------------------------
    csv_file.close()
    if cap is not None:
        cap.release()
    if not args.no_display:
        cv2.destroyAllWindows()

    # ---- summary -----------------------------------------------------------
    print(f"\nProcessed {frame_idx} frames total.")
    print(f"Per-frame metrics saved to: {csv_path}")
    if mask_dir:
        print(f"Mask images saved to: {mask_dir}")
    if mse_accum:
        mean_mse = np.mean(mse_accum)
        print(f"\nMean MSE over all frames (PyTorch as reference): {mean_mse:.6f}")
    else:
        print("\nNo tracking frames to compute MSE (only one frame was processed).")


if __name__ == "__main__":
    main()
