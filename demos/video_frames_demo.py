# Nanosam2 Video Frames Example
#
# Run inferences on the frames exported form a video.
#
# To run this script, create a new python environment (3.12) install all packages listed in the README.md file
# and add the "nanosam2" directory to your pythonpath.
# If you are using bash add the following line to your .bashrc
# export PYTHONPATH="<path-to>/nanosam2"



import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from nanosam2.sam2.build_sam import build_sam2_video_predictor
from pathlib import Path
import argparse

parser = argparse.ArgumentParser(description='Run benchmarks on different devices with different model optimizations.')
parser.add_argument("--config", type=str, default="sam2_hiera_s", help="The path to a sam2 config.")
parser.add_argument("--checkpoint", type=str, default="sam2_checkpoints/sam2.1_hiera_small.pt")
parser.add_argument('--video', help='Path to the frames (single jpg files) of your video.')
parser.add_argument('--out', type=str, help='Write output frames to this directory.', default="results")
args = parser.parse_args()

if not Path(args.video).exists():
    print(f"Video path {args.video} not found.")
    exit(1)

if not Path(args.checkpoint).exists():
    print(f"Checkpoint {args.video} not found.")
    exit(1)


# Configuration.
output_path = Path(args.out)
output_format = ".jpg"

# Create the output directory.
output_path.mkdir(parents=True, exist_ok=True)
(output_path / Path("#1")).mkdir(parents=True, exist_ok=True)
(output_path / Path("#2")).mkdir(parents=True, exist_ok=True)
(output_path / Path("#3")).mkdir(parents=True, exist_ok=True)
(output_path / Path("#4")).mkdir(parents=True, exist_ok=True)


# Device Selection.
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"using device: {device}")

if device.type == "cuda":
    # use bfloat16 for the entire notebook
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


# Build Predictor.
predictor = build_sam2_video_predictor(args.config, args.checkpoint, device=device)

# Helper Functions.
def show_mask(mask, ax, obj_id=None, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        cmap = plt.get_cmap("tab10")
        cmap_idx = 0 if obj_id is None else obj_id
        color = np.array([*cmap(cmap_idx)[:3], 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=200):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))


print("Scanning all JPEG frame names in video directory...")
frame_names = [
    p for p in os.listdir(args.video)
    if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
]
frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))
print(frame_names)

print("Loading video into nanosam2...")
inference_state = predictor.init_state(video_path=args.video)

predictor.reset_state(inference_state)

print("\n#1: Set two points and predict a mask...")
ann_frame_idx = 0  # the frame index we interact with
ann_obj_id = 1  # give a unique id to each object we interact with (it can be any integers)

# Add a positive click at (x, y).
# For labels, `1` means positive click and `0` means negative click
points = np.array([[770, 420], [750, 380]], dtype=np.float32)
labels = np.array([1,1], np.int32)
_, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
    inference_state=inference_state,
    frame_idx=ann_frame_idx,
    obj_id=ann_obj_id,
    points=points,
    labels=labels,
)

# Show the results on the current (interacted) frame.
plt.figure(figsize=(9, 6))
plt.title(f"frame {ann_frame_idx}")
plt.imshow(Image.open(os.path.join(args.video, frame_names[ann_frame_idx])))
show_points(points, labels, plt.gca())
show_mask((out_mask_logits[0] > 0.0).cpu().numpy(), plt.gca(), obj_id=out_obj_ids[0])
plt.savefig(f"{output_path}/#1/{args.config}_{ann_frame_idx}.{output_format}")

print("\n#2: Propagate across video with the two selected points...")
# Run propagation throughout the video.
video_segments = {}
for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
    video_segments[out_frame_idx] = {
        out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
        for i, out_obj_id in enumerate(out_obj_ids)
    }

# Render the segmentation results every few frames.
vis_frame_stride = 5
plt.close("all")
for out_frame_idx in range(0, len(frame_names), vis_frame_stride):
    plt.figure(figsize=(6, 4))
    plt.title(f"frame {out_frame_idx}")
    plt.imshow(Image.open(os.path.join(args.video, frame_names[out_frame_idx])))
    for out_obj_id, out_mask in video_segments[out_frame_idx].items():
        show_mask(out_mask, plt.gca(), obj_id=out_obj_id)
    plt.savefig(f"{output_path}/#2/{args.config}_{out_frame_idx}.{output_format}")


print("Reset predictor...")
predictor.reset_state(inference_state)

print("\n#3: Set a box and three refinement points to predict a mask...")
ann_frame_idx = 0  # the frame index we interact with
ann_obj_id = 4  # give a unique id to each object we interact with (it can be any integers)

# Add a positive click at (x, y).
points = np.array([[560, 350], [770, 420], [750, 380]], dtype=np.float32)
labels = np.array([1,1,1], np.int32)
# Box coordinates.
box = np.array([400, 320, 1100, 650], dtype=np.float32)
_, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
    inference_state=inference_state,
    frame_idx=ann_frame_idx,
    obj_id=ann_obj_id,
    points=points,
    labels=labels,
    box=box,
)

# Show the results on the current (interacted) frame.
plt.figure(figsize=(9, 6))
plt.title(f"frame {ann_frame_idx}")
plt.imshow(Image.open(os.path.join(args.video, frame_names[ann_frame_idx])))
show_box(box, plt.gca())
show_points(points, labels, plt.gca())
show_mask((out_mask_logits[0] > 0.0).cpu().numpy(), plt.gca(), obj_id=out_obj_ids[0])
plt.savefig(f"{output_path}/#3/{args.config}_{ann_frame_idx}.{output_format}")

print("\n#4: Propagate across the video with the box and the refinement points...")

# Run propagation throughout the video.
video_segments = {} 
for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
    video_segments[out_frame_idx] = {
        out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
        for i, out_obj_id in enumerate(out_obj_ids)
    }

# Render the segmentation results every "vis_frame_stride" frames.
vis_frame_stride = 5
plt.close("all")
for out_frame_idx in range(0, len(frame_names), vis_frame_stride):
    plt.figure(figsize=(6, 4))
    plt.title(f"frame {out_frame_idx}")
    plt.imshow(Image.open(os.path.join(args.video, frame_names[out_frame_idx])))
    for out_obj_id, out_mask in video_segments[out_frame_idx].items():
        show_mask(out_mask, plt.gca(), obj_id=out_obj_id)
    plt.savefig(f"{output_path}/#4/{args.config}_{out_frame_idx}.{output_format}")
