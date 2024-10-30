# nanosam2
This repository is inspired by https://github.com/NVIDIA-AI-IOT/nanosam and adapted for SAM2.1.
Although the inference speed of the SAM2.1 Hiera backbones is already quite fast on GPUs, it is still difficult to deploy on edge devices.
This repository aims to provide a more efficient alternative for SAM2.1 inference, with a focus on backbones that are smaller and faster to deploy.


## Dependencies and Prerequirements

To train and run the evaluation, the following packages are required:

```
pip install matplotlib torchvision tqdm hydra-core pycocotools requests iopath
```

### Image Preparation

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

#### Use Segment Anything Download Script

Instead of downloading and extracting the chunks manually, you can use the [download script](tools/download_sa_dataset.py). Download the textfile containing the list of all links and provide it to the download script. 

| Parameter | Description |
| --- | --- |
| -d  | Destination, if no destination is set, all files are downloaded to the current directory|
| -n  | number of chunks to download, if not set all chunks are downloaded |
| -e | Extract tar files |
| -r | Remove tar files after extracting |
```
python tools/download_sa_dataset.py links.txt -d <Destination> -n <Number of Chunks> -e
```

### Get original SAM2 checkpoints

Get the a original SAM2 checkpoint as described in the SAM2 repository. This checkpoint is required for the teaching network.

## Train

All experiments were conducted on a RTX 4090 GPU. So you might need to adjust the batch size for your GPU.

In the [train.py](nanosam2/tools/train.py.py).
 file, adjust the data set chunks you want to use for training (seach for `SA1Folder` in `train.py`).


### Train with resnet18 backbone and distill from sam2.1_hiera_s
```bash
python nanosam2/tools/train.py --images /path/to/images --output_dir results/sam2.1_hiera_s_resnet18 --model_name resnet18 --nanosam2_config nanosam2.1_resnet18 --sam2_config sam2.1_hiera_s --checkpoint sam2_checkpoints/sam2.1_hiera_small.pt --batch_size 16 --num_epochs 100 
```

### Train with MobileNetv3 Large backbone and distill from sam2.1_hiera_s

```bash
python nanosam2/tools/train.py --images /path/to/images --output_dir results/sam2.1_hiera_s_mobilenetV3_large --nanosam2_config nanosam2.1_mobilenet_v3_large --sam2_config sam2.1_hiera_s --teacher_checkpoint sam2_checkpoints/sam2.1_hiera_small.pt --batch_size 16 --num_epochs 100 --coco_root /path/to/coco --coco_ann /path/to/coco/annotations
```

## Evaluate on the validation set
Download Coco 2017 validation images and annotations from [here](https://cocodataset.org/#download), and evaluate the model:
```bash
python nanosam2/tools/eval_coco.py --checkpoint results/sam2.1_hiera_s_resnet18/checkpoint.pth --sam2_config nanosam2.1_resnet18 --output results/sam2.1_hiera_s_resnet18/coco_results.json
python nanosam2/tools/compute_eval_coco_metric.py results/sam2.1_hiera_s_resnet18/coco_results.json 
```


## Results FP32
Each backbone was trained for 10 epochs on 14 SA1 datasets, i.e. ~175k images.
| Backbone | num_epochs | mIoU  All | mIoU Small | mIoU Medium | mIoU Large |
| -------- | -------- | -------- | -------- | -------- | -------- |
| resnet18 | 10 | 0.69 | 0.62 | 0.73 | 0.76 |
| casvit_s | 10 | 0.71 | 0.64 | 0.75 | 0.78 |

## Results QAT
QAT was initialized with the result of FP32 training. Casvit fails to train for now.
Probably need to fuse some layers here. A subset of 30k frames was used and the networks are
trained for 20 epochs. Learning rate was set to 1e-4.

| Backbone | num_epochs | mIoU  All | mIoU Small | mIoU Medium | mIoU Large |
| -------- | -------- | -------- | -------- | -------- | -------- |
| resnet18_q | 20 | 0.67 | 0.61 | 0.71 | 0.72 |
| casvit_s_q | fails to train | - | - | - | - |


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
## Runtime benchmarks
Benchmarks executed on RTX3090 and AMD Ryzen 9 5900X 12-Core.
```bash
python nanosam2/tools/benchmark.py --config=sam2.1_hiera_t
```
### Resnet18
| device | dtype | batch_size | FPS (with torch.compile) |
| -------- | -------- | -------- | -------- |
| CPU | float32 | 1 | 4.9 (5.7) |
| CUDA | float32 | 1 | 150.2 (194.3) | 
| CUDA | float16 | 1 | 195.6 (368.3) | 
| CUDA | bfloat16 | 1 | 184.3 (352.8) | 

### Casvit_s
|  device | dtype | batch_size | FPS (with torch.compile)|
| -------- | -------- | -------- | -------- |
| CPU | float32 | 1 | 2.2 (5.0) |
| CUDA | float32 | 1 | 72.9 (111.2) | 
| CUDA | float16 | 1 | 81.4 (100.4) | 
| CUDA | bfloat16 | 1 | 78.7 (102.6) | 

### Hiera_s
|  device | dtype | batch_size | FPS (with torch.compile)|
| -------- | -------- | -------- | -------- |
| CPU | float32 | 1 | 1.1 (1.2) |
| CUDA | float32 | 1 | 28.5 (35.7) | 
| CUDA | float16 | 1 | 62.0 (93.3) | 
| CUDA | bfloat16 | 1 | 60.1 (92.4) | 

### Todo 

- [ ] Upload trained Resnet18 and Casvit_s backbones
- [ ] Add a video segmentation demo
- [ ] Improve instructions how to install the thing
