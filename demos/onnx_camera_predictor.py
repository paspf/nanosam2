# NanoSAM2 ONNX Live Demo
#
# Real-time object segmentation using the pure-ONNX NanoSAM2 pipeline.
# No PyTorch is required at inference time — only numpy and onnxruntime.
#
# Usage:
#   python demos/onnx_camera_predictor.py \
#       --model_dir model_exports \
#       --video 0 \
#       [--cfg ../sam2_configs/nanosam2.1_resnet18.yaml] \
#       [--checkpoint results/sam2.1_hiera_s_resnet18/checkpoint.pth] \
#       [--box x1,y1,x2,y2] [--box x1,y1,x2,y2 ...]   # pre-set boxes
#
# When --box is supplied the demo starts tracking immediately without
# requiring any interactive mouse input.  Each --box defines one object.
#
# The first time you run this script the model constants (embeddings, etc.)
# are extracted from the PyTorch checkpoint and saved as a .npz file next to
# the ONNX files.  Subsequent runs load the .npz directly and need no PyTorch.

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# Make sure the package root is importable when run as a script from any cwd
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from demos.demo_utils import (  # noqa: E402
    BoxDrawer, draw_masks, open_video_source, parse_boxes, read_next_frame,
)
from nanosam2.sam2.sam2_onnx_camera_predictor import build_onnx_camera_predictor  # noqa: E402


def _parse_args():
    parser = argparse.ArgumentParser(
        description="NanoSAM2 ONNX live demo — real-time object segmentation"
    )
    parser.add_argument(
        "--model_dir", default="model_exports",
        help="Directory with the exported ONNX files (default: model_exports)"
    )
    parser.add_argument(
        "--encoder_prefix", default="nanosam2-resnet18",
        help="ONNX file prefix, e.g. nanosam2-resnet18 (default: nanosam2-resnet18)"
    )
    parser.add_argument(
        "--video", default="0",
        help="Camera id (integer), path to a video file, or path to an image directory (default: 0)"
    )
    parser.add_argument(
        "--image_size", type=int, default=512,
        help="Model input resolution (default: 512)"
    )
    parser.add_argument(
        "--box", dest="boxes", action="append", metavar="x1,y1,x2,y2",
        help="Pre-set bounding box prompt as 'x1,y1,x2,y2' (pixel coords). "
             "Repeat to add multiple objects, e.g. --box 10,20,100,200 --box 300,50,450,180. "
             "When provided the demo starts tracking immediately without "
             "requiring any mouse input."
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = _parse_args()

    # ---- parse pre-set boxes -----------------------------------------------
    preset_boxes = []
    if args.boxes:
        try:
            preset_boxes = parse_boxes(args.boxes)
        except ValueError as e:
            print(f"[ERROR] {e}")
            sys.exit(1)

    # ---- open video / image directory --------------------------------------
    cap, image_dir_iter, src = open_video_source(args.video)

    # ---- build predictor ---------------------------------------------------
    print(f"Loading ONNX models from: {args.model_dir}")
    predictor = build_onnx_camera_predictor(
        onnx_dir=args.model_dir,
        encoder_prefix=args.encoder_prefix,
        image_size=args.image_size,
    )
    print("Predictor ready.")

    # ---- UI state ----------------------------------------------------------
    win = "NanoSAM2 ONNX"
    cv2.namedWindow(win)
    drawer = BoxDrawer()
    cv2.setMouseCallback(win, drawer.mouse_cb)

    if_init = False
    obj_counter = 0
    frametimes = []

    print()
    if preset_boxes:
        print(f"Starting with {len(preset_boxes)} pre-set box(es) -- tracking begins immediately.")
    else:
        print("Controls:")
        print("  Draw a bounding box with the LEFT MOUSE BUTTON to add an object.")
    print("  R -- reset and re-initialise on the next frame.")
    print("  Q / ESC -- quit.")
    print()

    while True:
        bgr = read_next_frame(cap, image_dir_iter, src)
        if bgr is None:
            break

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        display = bgr.copy()

        # ---- initialise on the first usable frame --------------------------
        if not if_init:
            predictor.load_first_frame(rgb)

            if preset_boxes:
                # Apply all pre-set boxes immediately — no user interaction needed
                for x1, y1, x2, y2 in preset_boxes:
                    obj_counter += 1
                    box = np.array([[x1, y1], [x2, y2]], dtype=np.float32)
                    _, obj_ids, mask_logits = predictor.add_new_points_or_box(
                        frame_idx=0,
                        obj_id=obj_counter,
                        box=box,
                        normalize_coords=True,
                    )
                display  = draw_masks(display, obj_ids, mask_logits)
                if_init  = True
                cv2.imshow(win, display)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord('q'), 27):
                    break
                continue
            else:
                # Interactive: wait for the user to draw a box
                hint = display.copy()
                cv2.putText(hint, "Draw a bounding box to start tracking",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                            (0, 255, 0), 2, cv2.LINE_AA)
                drawer.draw_on(hint)
                cv2.imshow(win, hint)

                key = cv2.waitKey(1) & 0xFF
                if key in (ord('q'), 27):
                    break
                if key == ord('r'):
                    predictor.reset_state()
                    drawer.reset()
                    obj_counter = 0

                if drawer.done:
                    box_coords = drawer.get_box()
                    if box_coords is not None:
                        x1, y1, x2, y2 = box_coords
                        obj_counter += 1
                        box = np.array([[x1, y1], [x2, y2]], dtype=np.float32)
                        _, obj_ids, mask_logits = predictor.add_new_points_or_box(
                            frame_idx=0,
                            obj_id=obj_counter,
                            box=box,
                            normalize_coords=True,
                        )
                        display = draw_masks(display, obj_ids, mask_logits)
                        if_init = True
                        drawer.reset()
                continue

        # ---- check for additional box prompts while tracking ---------------
        if drawer.done:
            box_coords = drawer.get_box()
            print(f"Box drawn: {box_coords}" if box_coords is not None else "Box too small, ignored.")
            if box_coords is not None:
                x1, y1, x2, y2 = box_coords
                obj_counter += 1
                # Reset and re-initialise to add a new object
                predictor.reset_state()
                obj_counter_before = obj_counter
                predictor.load_first_frame(rgb)
                box = np.array([[x1, y1], [x2, y2]], dtype=np.float32)
                _, obj_ids, mask_logits = predictor.add_new_points_or_box(
                    frame_idx=0,
                    obj_id=obj_counter,
                    box=box,
                    normalize_coords=True,
                )
                display = draw_masks(display, obj_ids, mask_logits)
                drawer.reset()
                frametimes.clear()
                cv2.imshow(win, display)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord('q'), 27):
                    break
                continue

        # ---- track ---------------------------------------------------------
        t0 = time.perf_counter()
        obj_ids, mask_logits = predictor.track(rgb)
        dt = time.perf_counter() - t0
        frametimes.append(dt)

        display = draw_masks(display, obj_ids, mask_logits)

        # HUD
        fps_str = f"FPS: {1.0/dt:.1f}" if dt > 0 else "FPS: --"
        cv2.putText(display, fps_str, (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(display, "R=reset  Q=quit",
                    (10, display.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 1, cv2.LINE_AA)
        drawer.draw_on(display)

        cv2.imshow(win, display)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            break
        if key == ord('r'):
            predictor.reset_state()
            drawer.reset()
            obj_counter = 0
            if_init = False
            frametimes.clear()
            # preset_boxes will be re-applied on the next init frame automatically

    if cap is not None:
        cap.release()
    cv2.destroyAllWindows()

    if frametimes:
        avg = np.mean(frametimes)
        print(f"\nAverage inference time : {avg*1000:.1f} ms")
        print(f"Average FPS            : {1.0/avg:.1f}")


if __name__ == "__main__":
    main()
