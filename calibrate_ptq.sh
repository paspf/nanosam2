python nanosam2/tools/ptq.py --images /media/steffen/Data/Downloads/SA1/raw \
 --fp32_checkpoint_for_ptq results/sam2.1_hiera_s_resnet18/checkpoint.pth \
 --sam2_config sam2.1_hiera_s \
 --teacher_checkpoint sam2_checkpoints/sam2.1_hiera_small.pt \
 --nanosam2_config nanosam2.1_resnet18 \
 --output_dir results/sam2.1_hiera_s_resnet18_ptq
