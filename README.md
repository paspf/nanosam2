# nanosam2
This repository is inspired by https://github.com/NVIDIA-AI-IOT/nanosam and adapted for SAM2.1.
Although the inference speed of the SAM2.1 Hiera backbones is already quite fast on GPUs, it is still difficult to deploy on edge devices.
This repository aims to provide a more efficient alternative for SAM2.1 inference, with a focus on backbones that are smaller and faster to deploy.

## Prepare images
Download the chunks from the [SA1 dataset](https://ai.meta.com/datasets/segment-anything-downloads/) that you want to train on, and put them in a folder. For example:
```
data/
    sa_000000/
        sa_1.jpg
        sa_2.jpg
        ...
    sa_000001/
        sa_x.jpg
        ...
```
You can then point the training script to this folder.

## Train with different backbone
All experiments were conducted on a RTX 4090 GPU. So you might need to adjust the batch size for your GPU.

### Train with resnet18 backbone and distill from sam2.1_hiera_s
```bash
python nanosam2/tools/train_from_images.py --images /path/to/images --output_dir results/sam2.1_hiera_s_resnet18 --model_name resnet18 --nanosam2_config nanosam2.1_resnet18 --sam2_config sam2.1_hiera_s --checkpoint sam2_checkpoints/sam2.1_hiera_small.pt --batch_size 16 --num_epochs 100 
```

### Evaluate on the validation set
Download Coco 2017 validation images and annotations from [here](https://cocodataset.org/#download), and evaluate the model:
```bash
python nanosam2/tools/eval_coco.py --checkpoint results/sam2.1_hiera_s_resnet18/checkpoint.pth --sam2_config nanosam2.1_resnet18 --output results/sam2.1_hiera_s_resnet18/coco_results.json
python nanosam2/tools/compute_eval_coco_metric.py results/sam2.1_hiera_s_resnet18/coco_results.json 
```


### Results
Each backbone was trained for 10 epochs on 14 SA1 datasets, i.e. ~175k images.
| Backbone | num_epochs | mIoU  All | mIoU Small | mIoU Medium | mIoU Large |
| -------- | -------- | -------- | -------- | -------- | -------- |
| resnet18 | 10 | 0.69 | 0.62 | 0.73 | 0.76 |
| casvit_s | 10 | 0.71 | 0.64 | 0.75 | 0.78 |

### Testing SAM2.1 Hiera backbones with different resolutions
Change the image size in the config file, and evaluate on coco val2017. It seems that performance is sensitive to the image size, and the best results were obtained with the training resolution of 1024. So I would not recommend to use smaller resolutions for distillation. 

Results:
| Backbone | Image Size | mIoU All | mIoU Small | mIoU Medium | mIoU Large |
| -------- | -------- | -------- | -------- | -------- | -------- |
| hiera_s | 512 | 68.4 | 57.0 | 75.6 | 77.9 |
| hiera_s | 768 | 73.7 | 65.9 | 78.7 | 79.9 |
| hiera_s | 1024 | 75.0 | 67.9 | 79.6 | 80.6 |
| hiera_s | 1440 | 74.8 | 67.9 | 79.1 | 80.6 |

You can reproduce these results by first predicting the masks on the coco val2017 images:
```bash
python nanosam2/tools/eval_coco.py --checkpoint sam2_checkpoints/sam2.1_hiera_small.pt --sam2_config sam2.1_hiera_s --output results/sam2.1_hiera_s/coco_results.json --coco_root PATH_TO_COCO_VAL2017 --coco_ann PATH_TO_COCO_ANNOTATIONS
```
and then compute the mIoU:
```bash
python nanosam2/tools/compute_eval_coco_metric.py results/sam2.1_hiera_s/coco_results.json --size all (or small, medium, large)
```

### Todo 
[ ] Upload trained Resnet18 and Casvit_s backbones
[ ] Add a video segmentation demo
