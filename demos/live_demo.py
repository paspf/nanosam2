# Live Demo
# Based on "https://github.com/Gy920/segment-anything-2-real-time/blob/main/demo/demo.py".


import torch
import numpy as np
import cv2
import argparse
from nanosam2.sam2.build_sam import build_sam2_camera_predictor
import time

parser = argparse.ArgumentParser(description='Run benchmarks on different devices with different model optimizations.')
parser.add_argument("--config", type=str, default="sam2_hiera_s", help="The path to a sam2 config.")
parser.add_argument("--checkpoint", type=str, default="sam2_checkpoints/sam2.1_hiera_small.pt")
parser.add_argument('--video', default=0, help='Path to a video or a camera id, default: 0')
args = parser.parse_args()

device = "cuda"
if device == "cuda":
    # use bfloat16 for the entire notebook
    torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

    if torch.cuda.get_device_properties(0).major >= 8:
        # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

predictor = build_sam2_camera_predictor(args.config, args.checkpoint, device=torch.device(device))
frametimes = []

def _compile_model_blocks(model, model_settings:list, compile_backend):
    print("Compiling Model...")
    if model_settings[0]: # image_encoder
        model.image_encoder = torch.compile(model.image_encoder, backend=compile_backend, dynamic=False)
    if model_settings[1]: # memory_attention
        model.memory_attention = torch.compile(model.memory_attention, backend=compile_backend)
    if model_settings[2]: # sam_mask_decoder
        model.sam_mask_decoder = torch.compile(model.sam_mask_decoder, backend=compile_backend)
    if model_settings[3]: # sam_prompt_encoder
        model.sam_prompt_encoder = torch.compile(model.sam_prompt_encoder, backend=compile_backend)
    if model_settings[4]: # memory_encoder
        model.memory_encoder = torch.compile(model.memory_encoder, backend=compile_backend)
    print("Compile finished.")
    return model


#predictor = _compile_model_blocks(predictor, [True, False, False, False, False], "inductor")
cap = cv2.VideoCapture(args.video)

if_init = False


while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    width, height = frame.shape[:2][::-1]
    if not if_init:

        predictor.load_first_frame(frame)
        if_init = True

        ann_frame_idx = 0  # the frame index we interact with
        ann_obj_id = 1  # give a unique id to each object we interact with (it can be any integers)
        # Let's add a positive click at (x, y) = (210, 350) to get started


        ##! add points, `1` means positive click and `0` means negative click
        # points = np.array([[660, 267]], dtype=np.float32)
        # labels = np.array([1], dtype=np.int32)

        # _, out_obj_ids, out_mask_logits = predictor.add_new_prompt(
        #     frame_idx=ann_frame_idx, obj_id=ann_obj_id, points=points, labels=labels
        # )

        ## ! add bbox
        bbox = np.array([[600, 214], [765, 286]], dtype=np.float32)
        _, out_obj_ids, out_mask_logits = predictor.add_new_prompt(
            frame_idx=ann_frame_idx, obj_id=ann_obj_id, bbox=bbox
        )

        ##! add mask
        # mask_img_path="../notebooks/masks/aquarium/aquarium_mask.png"
        # mask = cv2.imread(mask_img_path, cv2.IMREAD_GRAYSCALE)
        # mask = mask / 255

        # _, out_obj_ids, out_mask_logits = predictor.add_new_mask(
        #     frame_idx=ann_frame_idx, obj_id=ann_obj_id, mask=mask
        # )

    else:
        start = time.perf_counter()
        out_obj_ids, out_mask_logits = predictor.track(frame)
        mid = time.perf_counter()
        all_mask = np.zeros((height, width, 1), dtype=np.uint8)
        # print(all_mask.shape)
        for i in range(0, len(out_obj_ids)):
            out_mask = (out_mask_logits[i] > 0.0).permute(1, 2, 0).cpu().numpy().astype(
                np.uint8
            ) * 255

            all_mask = cv2.bitwise_or(all_mask, out_mask)

        all_mask = cv2.cvtColor(all_mask, cv2.COLOR_GRAY2RGB)
        frame = cv2.addWeighted(frame, 1, all_mask, 0.5, 0)
        end = time.perf_counter()
        prediction_time = (mid-start)
        drawing_time = (end-mid)
        frametimes.append(prediction_time)
        # print(f"prediction_time: {prediction_time} | drawing time: {drawing_time}")
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    cv2.imshow("frame", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
print("Video ended.")
print(f"Average runtime per frame: {np.average(frametimes)}s | Average FPS: {1/np.average(frametimes)}")