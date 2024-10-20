# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import json
from torchvision.datasets import CocoDetection
import numpy as np
import matplotlib.pyplot as plt
import tqdm
import torch
import argparse
from nanosam2.sam2.build_sam import build_sam2
from nanosam2.sam2.sam2_image_predictor import SAM2ImagePredictor
import os

def predict_box(predictor, image, box, set_image=True):

    if set_image:
        predictor.set_image(image)

    points = np.array([
        [box[0], box[1]],
        [box[2], box[3]]
    ])
    point_labels = np.array([2, 3])
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        mask, iou_preds, low_res_mask = predictor.predict(
            point_coords=points,
            point_labels=point_labels
        )

    mask = mask[iou_preds.argmax()] > 0
    
    return mask

    
def box_xywh_to_xyxy(box):
    return [box[0], box[1], box[0] + box[2], box[1] + box[3]]


def draw_box(box):
    x = [box[0], box[0], box[2], box[2], box[0]]
    y = [box[1], box[3], box[3], box[1], box[1]]
    plt.plot(x, y, 'g-')


def iou(mask_a, mask_b):
    intersection = np.count_nonzero(mask_a & mask_b)
    union = np.count_nonzero(mask_a | mask_b)
    return intersection / union


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--coco_root", type=str, default="/media/steffen/Data/Downloads/Coco/val2017")
    parser.add_argument("--coco_ann", type=str, default="/media/steffen/Data/Downloads/Coco/annotations/instances_val2017.json")
    parser.add_argument("--checkpoint", type=str, default="results/sam2.1_hiera_s_resnet18/checkpoint.pth",  help="The path to a checkpoint to resume training.")
    parser.add_argument("--sam2_config", type=str, default="nanosam2.1_resnet18", help="Sam2 config name, e.g. sam2.1_hiera_s")
    parser.add_argument("--output", type=str, default="data/coco_results.json")
    args = parser.parse_args()
        
    dataset = CocoDetection(
        root=args.coco_root,
        annFile=args.coco_ann)

    sam21 = build_sam2(args.sam2_config, ckpt_path=args.checkpoint,
                       device="cuda", mode="eval", 
                       apply_postprocessing=False, 
                       load_image_encoder=True)

    predictor = SAM2ImagePredictor(sam21)

    results = []

    for i in tqdm.tqdm(range(len(dataset))):

        image, anns = dataset[i]

        for j, ann in enumerate(anns):

            id = ann['id']
            area = ann['area']
            category_id = ann['category_id']
            iscrowd = ann['iscrowd']
            image_id = ann['image_id']
            box = box_xywh_to_xyxy(ann['bbox'])
            mask = dataset.coco.annToMask(ann)
            mask_coco = (mask > 0)
            mask_sam = predict_box(predictor, image, box, set_image=(j==0))
            # plot masks over rgb image, both the ground truth and the predicted mask
            # plot side by side
            plt.figure(figsize=(10, 5))
            plt.subplot(1, 2, 1)
            plt.imshow(image)
            plt.imshow(mask_coco, alpha=0.5)
            draw_box(box)
            plt.subplot(1, 2, 2)
            plt.imshow(image)
            plt.imshow(mask_sam, alpha=0.5)
            draw_box(box)
            plt.show()
            result = {
                "id": ann['id'],
                "area": ann['area'],
                "category_id": ann['category_id'],
                "iscrowd": ann['iscrowd'],
                "image_id": ann["image_id"],
                "box": box,
                "iou": iou(mask_sam, mask_coco)
            }

            results.append(result)
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    with open(args.output, 'w') as f:
        json.dump(results, f)
