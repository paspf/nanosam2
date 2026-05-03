# NanoSAM2 PyTorch Live Demo
#
# Real-time object segmentation using the PyTorch NanoSAM2 pipeline.
#
# Usage:
#   python demos/torch_camera_predictor.py \
#       --config nanosam2.1_resnet18 \
#       --checkpoint results/sam2.1_hiera_s_resnet18/checkpoint.pth \
#       --video 0 \
#       [--device cuda] \
#       [--box x1,y1,x2,y2] [--box x1,y1,x2,y2 ...]   # pre-set boxes
#
# When --box is supplied the demo starts tracking immediately without
# requiring any interactive mouse input.  Each --box defines one object.
# Without --box the first frame is paused and you draw a bounding box with
# the left mouse button to initialise tracking.
#
# Based on https://github.com/Gy920/segment-anything-2-real-time/blob/main/demo/demo.py

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

# Make sure the package root is importable when run as a script from any cwd
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from demos.demo_utils import (  # noqa: E402
    BoxDrawer, draw_masks, open_video_source, parse_boxes,
    read_next_frame, torch_masks_to_numpy,
)
from nanosam2.sam2.build_sam import build_sam2_camera_predictor  # noqa: E402


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def _parse_args():
    parser = argparse.ArgumentParser(
        description="NanoSAM2 PyTorch live demo -- real-time object segmentation"
    )
    parser.add_argument(
        "--config", default="nanosam2.1_resnet18",
        help="SAM2 config name or path (default: nanosam2.1_resnet18)"
    )
    parser.add_argument(
        "--checkpoint",
        default="results/sam2.1_hiera_s_resnet18/checkpoint.pth",
        help="Path to the PyTorch model checkpoint"
    )
    parser.add_argument(
        "--video", default="0",
        help="Camera id (integer), path to a video file, or path to an image directory (default: 0)"
    )
    parser.add_argument(
        "--device", default="cpu",
        help="Inference device: cpu or cuda (default: cpu)"
    )
    parser.add_argument(
        "--box", dest="boxes", action="append", metavar="x1,y1,x2,y2",
        help="Pre-set bounding box prompt as x1,y1,x2,y2 (pixel coords). "
             "Repeat to add multiple objects. "
             "When provided the demo starts tracking immediately without mouse input."
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# PyTorch model compilation helper
# ---------------------------------------------------------------------------

def _compile_model_blocks(model, which: list, backend: str = "inductor"):
    # Optionally torch.compile individual model sub-blocks.
    if not any(which):
        return model
    print("Compiling model blocks...")
    names = ["image_encoder", "memory_attention", "sam_mask_decoder",
             "sam_prompt_encoder", "memory_encoder"]
    for flag, name in zip(which, names):
        if flag:
            print(f"  - {name}")
            setattr(model, name,
                    torch.compile(getattr(model, name), backend=backend,
                                  dynamic=name == "memory_attention"))
    print("Compilation done.")
    return model


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

    # ---- configure device --------------------------------------------------
    device = args.device
    if device == "cuda":
        torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

    # ---- build predictor ---------------------------------------------------
    print(f"Loading model: {args.config}")
    predictor = build_sam2_camera_predictor(
        args.config, args.checkpoint, device=torch.device(device)
    )
    # Compile if desired (all False by default -- change to enable)
    predictor = _compile_model_blocks(
        predictor, [False, False, False, False, False]
    )
    print("Predictor ready.")

    # ---- open video / image directory --------------------------------------
    cap, image_dir_iter, src = open_video_source(args.video)

    # ---- UI state ----------------------------------------------------------
    win = "NanoSAM2 PyTorch"
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
                # Apply all pre-set boxes immediately -- no user interaction needed
                for x1, y1, x2, y2 in preset_boxes:
                    obj_counter += 1
                    bbox = np.array([[x1, y1], [x2, y2]], dtype=np.float32)
                    _, obj_ids, mask_logits = predictor.add_new_prompt(
                        frame_idx=0,
                        obj_id=obj_counter,
                        bbox=bbox,
                    )
                display  = draw_masks(display, obj_ids,
                                      torch_masks_to_numpy(mask_logits))
                if_init  = True
                cv2.imshow(win, display)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
            else:
                # Interactive: wait for the user to draw a box
                hint = display.copy()
                cv2.putText(hint, "Draw a bounding box to start tracking",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                            (0, 255, 0), 2, cv2.LINE_AA)
                drawer.draw_on(hint)
                cv2.imshow(win, hint)

                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key == ord("r"):
                    predictor.reset_state()
                    drawer.reset()
                    obj_counter = 0

                if drawer.done:
                    box_coords = drawer.get_box()
                    if box_coords is not None:
                        x1, y1, x2, y2 = box_coords
                        obj_counter += 1
                        bbox = np.array([[x1, y1], [x2, y2]], dtype=np.float32)
                        _, obj_ids, mask_logits = predictor.add_new_prompt(
                            frame_idx=0,
                            obj_id=obj_counter,
                            bbox=bbox,
                        )
                        display = draw_masks(display, obj_ids,
                                             torch_masks_to_numpy(mask_logits))
                        if_init = True
                        drawer.reset()
                continue

        # ---- check for additional box prompts while tracking ---------------
        if drawer.done:
            box_coords = drawer.get_box()
            if box_coords is not None:
                x1, y1, x2, y2 = box_coords
                obj_counter += 1
                # Reset and re-initialise with the new box on the current frame
                predictor.reset_state()
                predictor.load_first_frame(rgb)
                bbox = np.array([[x1, y1], [x2, y2]], dtype=np.float32)
                _, obj_ids, mask_logits = predictor.add_new_prompt(
                    frame_idx=0,
                    obj_id=obj_counter,
                    bbox=bbox,
                )
                display = draw_masks(display, obj_ids,
                                     torch_masks_to_numpy(mask_logits))
                drawer.reset()
                frametimes.clear()
                cv2.imshow(win, display)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                continue

        # ---- track ---------------------------------------------------------
        t0 = time.perf_counter()
        out_obj_ids, out_mask_logits = predictor.track(rgb)
        dt = time.perf_counter() - t0
        frametimes.append(dt)

        display = draw_masks(display, out_obj_ids,
                             torch_masks_to_numpy(out_mask_logits))

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
        if key in (ord("q"), 27):
            break
        if key == ord("r"):
            predictor.reset_state()
            drawer.reset()
            obj_counter = 0
            if_init     = False
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
