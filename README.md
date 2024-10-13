# nanosam2


## Train with different backbone
All experiments were conducted on a RTX 4090 GPU. So you might need to adjust the batch size for your GPU.

### Train with resnet18 backbone and distill from sam2.1_hiera_s
```bash
python nanosam2/tools/train_from_images.py --images /path/to/images --output_dir results/sam2.1_hiera_s_resnet18 --model_name resnet18 --nanosam2_config nanosam2.1_resnet18 --sam2_config sam2.1_hiera_s --checkpoint sam2_checkpoints/sam2.1_hiera_small.pt --batch_size 16 --num_epochs 100 
```

