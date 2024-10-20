import tqdm
import torch
import argparse
from nanosam2.datasets.image_folder import SA1Folder
import os
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from nanosam2.sam2.build_sam import build_sam2

if __name__ == "__main__":
        
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", required=True, type=str, help="The path to images to use for distillation")
    parser.add_argument("--output_dir", type=str, help="The directory to store checkpoints and training visualizations.")

    parser.add_argument("--batch_size", type=int, default=12, help="The batch size.")
    parser.add_argument("--num_workers", type=int, default=8, help="The number of data loader workers.")
    

    parser.add_argument("--teacher_checkpoint", type=str, default="sam2_checkpoints/sam2.1_hiera_large.pt", 
                        help="The path to a checkpoint to resume training.")
    parser.add_argument("--sam2_config", type=str, default="sam2.1_hiera_l", help="The path to a checkpoint to resume training.")
    parser.add_argument("--nanosam2_config", type=str, default="nanosam2.1_casvit", help="The path to a checkpoint to resume training.")

    parser.add_argument("--fp32_checkpoint_for_ptq", default=None, help="The path to a nanosam FP32 checkpoint to use for PTQ.")
    args = parser.parse_args()

    if args.fp32_checkpoint_for_ptq is None and args.qat:
        raise ValueError("You must provide a FP32 checkpoint for PTQ!")

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    sam21 = build_sam2(args.sam2_config, ckpt_path=args.teacher_checkpoint, 
            device="cuda", mode="eval", apply_postprocessing=False, load_image_encoder=True)
    nanosam21 = build_sam2(args.nanosam2_config, ckpt_path=args.teacher_checkpoint, 
            device="cuda", mode="eval", apply_postprocessing=False, load_image_encoder=False)

    if nanosam21.image_size != sam21.image_size:
        raise ValueError(f"Image size of student and teacher must be the same, got {nanosam21.image_size} and {sam21.image_size}")

    dataset = SA1Folder([0, 1], args.images, resolution=sam21.image_size)

    loader = DataLoader(dataset, shuffle=True, 
                        batch_size=args.batch_size, num_workers=args.num_workers)

    checkpoint_path = os.path.join(args.output_dir, "checkpoint_ptq.pth")
    start_epoch = 0

    print("Enabling PTQ for the image encoder")
    nanosam21.eval()
    nanosam21.image_encoder.qconfig = torch.ao.quantization.get_default_qat_qconfig('x86')
    torch.ao.quantization.prepare(nanosam21.image_encoder, inplace=True)
    
    for idx, image in enumerate(tqdm.tqdm(iter(loader))):
        image = image.cuda()
        if len(image) != args.batch_size:
            continue
        image_cnn = F.interpolate(image, (sam21.image_size, sam21.image_size), mode="area")

        nanosam21.image_encoder(image_cnn)

    # nanosam21.image_encoder = torch.ao.quantization.convert(nanosam21.image_encoder)
    torch.save({"model": nanosam21.state_dict()}, checkpoint_path)

