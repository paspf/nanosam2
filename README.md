# nanosam2

NanoSAM2 is a lightweight, deployment-friendly implementation of [SAM2.1](https://github.com/facebookresearch/sam2),
inspired by [NanoSAM](https://github.com/NVIDIA-AI-IOT/nanosam).

While the original SAM2.1 Hiera backbones run well on high-end GPUs, they remain difficult to deploy on
edge devices or in environments where PyTorch is unavailable.
NanoSAM2 addresses this with two complementary strategies:

- **Lightweight backbones** — drop-in replacements for the Hiera trunk (ResNet18, MobileNetV3-Large,
  CasViT-S) that are significantly smaller and faster, trained via knowledge distillation from SAM2.1.
- **Pure ONNX runtime** — all model components (image encoder, SAM heads, memory encoder, memory
  attention) can be exported to ONNX and run entirely with `onnxruntime` and `numpy`.
  No PyTorch installation is required at inference time.

## Installation

### Dependencies and Prerequirements

 - Create a new Python 3.12+ environment and clone the repository.
 - Install the dependencies listed below:

```
pip install matplotlib torchvision tqdm hydra-core pycocotools requests iopath opencv-python
```

 - Install the repository as editable package `pip install -e .`

### Download checkpoints

You can find and download pretrained nanosam2 checkpoints [here](https://drive.google.com/drive/folders/15wApVHwqJGunjDP_cx5YZDCTEKliOMCQ?usp=sharing). Each backbone was trained for 10 epochs on 14 SA1 datasets, i.e. ~175k images. 

## Inference Demos

All inference demos are executed on the [Bedroom](https://github.com/facebookresearch/sam2/blob/2b90b9f5ceec907a1c18123530e92e794ad901a4/notebooks/videos/bedroom.mp4) video file, shared in the original [sam2](https://github.com/facebookresearch/sam2) repository. 

### Video Predictor

Load all frames of a video at once into Nanosam2 and perform tracking of objects from any frame. To run the video demo you have to obtain all frames of the video as `.jpg` file. Place all `.jpg` files
in the same directory and pass the directory to `video_frames_demo.py`. 

Extracting all frames of a video uns FFmpeg:
```
ffmpeg -i video.mp4 05%d.jpg
```

Calling the inference script with ResNet18 backend:

```
python demos/torch_video_predictor.py --config nanosam2.1_resnet18.yaml --checkpoint results/sam2.1_hiera_s_resnet18/checkpoint.pth --video <directory with exported video frames>
```


### Camera Predictor

Stream a video (from a camera or video file source) frame by frame into Nanosam2. Start object tracking from any frame in the stream.


For ResNet18 backend with two box prompts:

```bash
python demos/torch_camera_predictor.py --config nanosam2.1_resnet18.yaml --checkpoint results/sam2.1_hiera_s_resnet18/checkpoint.pth --video videos/bedroom --box 230,134,294,219 --box 353,11,451,122
```

For ResNet18 backend, use mouse to set box prompt:

```bash
python demos/torch_camera_predictor.py --config nanosam2.1_resnet18.yaml --checkpoint results/sam2.1_hiera_s_resnet18/checkpoint.pth --video <Video device id or path to mp4 file>
```

For MobileNetV3 backend:

```bash
python demos/torch_camera_predictor.py --config nanosam2.1_mobilenet_v3_large.yaml --checkpoint results/sam2.1_hiera_s_mobilenetV3_large/checkpoint.pth --video <Video device id or path to mp4 file>
```

For original hiera backend:

```bash
python demos/torch_camera_predictor.py --config sam2.1_hiera_s.yaml --checkpoint sam2_checkpoints/sam2.1_hiera_small.pt --video <Video device id or path to mp4 file>
```



## ONNX Export and Pure-ONNX Inference

NanoSAM2 can be fully exported to ONNX so that **no PyTorch is required at inference time** — only `numpy` and `onnxruntime`.
The model is split into four independent ONNX files: image encoder, SAM heads, memory encoder, and memory attention.

### Additional dependencies

```bash
pip install onnx onnxruntime onnxsim
```

### Export model blocks

Run the exporter from the `nanosam2/` directory:

```bash
# ResNet18 backbone at 512×512 (trained resolution)
python tools/onnx_export.py --encoder_type resnet18 --export all --output_path model_exports --opset 18 --img_shape 3 512 512

# Original SAM2.1 Hiera-Small backbone at 1024×1024 (trained resolution)
python tools/onnx_export.py \
    --encoder_type hiera_small \
    --export all \
    --output_path model_exports \
    --opset 18 \
    --img_shape 3 1024 1024
```

Individual blocks can be exported by replacing `all` with one of:
`image-encoder`, `image-encoder-trunk`, `sam-heads`, `memory-encoder`, `memory-attention`.

This produces files named `<model>-<block>-sa1-v01-op18.onnx` in `model_exports/`.

### Inference Demos (ONNX Runtime)

The `demos/onnx_camera_predictor.py` demo streams from a camera, a video file, or a directory of images
and tracks objects selected with a bounding-box prompt.

**Interactive mode** — draw a box with the mouse on the first frame:

```bash
# Camera
python demos/onnx_camera_predictor.py --model_dir model_exports --video 0

# Video file
python demos/onnx_camera_predictor.py --model_dir model_exports --video path/to/video.mp4

# Directory of images (sorted alphabetically, loops)
python demos/onnx_camera_predictor.py --model_dir model_exports --video videos/bedroom
```

**Pre-set box mode** — tracking starts immediately, no mouse input needed.
Pass one or more `--box x1,y1,x2,y2` arguments (pixel coordinates on the original frame):

```bash
# Single object
python demos/onnx_camera_predictor.py --model_dir model_exports --video videos/bedroom --box 230,134,294,219

# Two objects
python demos/onnx_camera_predictor.py --model_dir model_exports --video videos/bedroom --box 230,134,294,219 --box 353,11,451,122
```

**Using the Hiera-Small backbone** at 1024×1024:

```bash
python demos/onnx_live_demo.py \
    --model_dir model_exports \
    --encoder_prefix sam2.1_small \
    --image_size 1024 \
    --video 0
```

**Key controls during the demo:**

| Key | Action |
|-----|--------|
| Left mouse button (drag) | Draw a bounding box to add / re-initialise an object |
| `R` | Reset tracking, re-initialise on the next frame |
| `Q` / `ESC` | Quit |

### Todo 

- [x] Upload trained Resnet18, mobilenet and Casvit_s backbones
- [x] Add a video segmentation demo
- [x] Improve instructions how to install the thing
- [x] ONNX export for all model blocks
- [x] Pure-ONNX camera predictor (no PyTorch at inference)
- [x] ONNX live demo with interactive and pre-set box prompts

### Citing
If you find this repository useful for your work consider citing it :)
```bibtex
    @misc{UrbanPfeifferGithubRepo,
      author       = {Steffen Urban and Pascal Pfeiffer},
      title        = {nanosam2},
      year         = {2024},
      howpublished = {\url{https://github.com/urbste/nanosam2}},
      note         = {Accessed: 2024-11-21}
    }
```


## Training Prerequirements

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
You can find pretrained backbones (for ~10epochs) [here](https://drive.google.com/drive/folders/15wApVHwqJGunjDP_cx5YZDCTEKliOMCQ?usp=sharing).

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
| mobilenetV3 large | 20 | 0.68 | 0.60 | 0.71 | 0.75 |

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

---

