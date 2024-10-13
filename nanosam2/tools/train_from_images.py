import tqdm
import torch
import argparse
from nanosam2.datasets.image_folder import SA1Folder
import os
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from nanosam2.sam2.build_sam import build_sam2

loss_weight = [1.0, 1/4, 1/8]

if __name__ == "__main__":
        
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", type=str, help="The path to images to use for distillation")
    parser.add_argument("--output_dir", type=str, help="The directory to store checkpoints and training visualizations.")
    parser.add_argument("--model_name", type=str, default="casvit", help="The NanoSAM2 model name.")
    parser.add_argument("--student_size", type=int, default=1024, help="The size of image to feed to the student during distillation.")
    parser.add_argument("--num_images", type=int, default=None, help="Limit the number of images per epoch.")
    parser.add_argument("--num_epochs", type=int, default=100, help="The number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=12, help="The batch size.")
    parser.add_argument("--num_workers", type=int, default=8, help="The number of data loader workers.")
    parser.add_argument("--learning_rate", type=float, default=3e-4, help='The learning rate.')

    parser.add_argument("--checkpoint", type=str, default="sam2_checkpoints/sam2.1_hiera_small.pt", help="The path to a checkpoint to resume training.")
    parser.add_argument("--sam2_config", type=str, default="sam2.1_hiera_s", help="The path to a checkpoint to resume training.")
    parser.add_argument("--nanosam2_config", type=str, default="nanosam2.1_casvit", help="The path to a checkpoint to resume training.")
    parser.add_argument("--loss", type=str, default="huber", choices=["huber", "l1", "mse"], help="The loss function to use for distillation.")
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    sam21 = build_sam2(args.sam2_config, ckpt_path=args.checkpoint, 
            device="cuda", mode="eval", apply_postprocessing=False, load_image_encoder=True)
    nanosam21 = build_sam2(args.nanosam2_config, ckpt_path=args.checkpoint, 
            device="cuda", mode="eval", apply_postprocessing=False, load_image_encoder=False)

    if args.loss == "huber":
        loss_function = F.huber_loss
    elif args.loss == "l1":
        loss_function = F.l1_loss
    elif args.loss == "mse":
        loss_function = F.mse_loss
    else:
        raise RuntimeError(f"Unsupported loss function {args.loss}")

    optimizer = torch.optim.Adam(nanosam21.image_encoder.parameters(), lr=args.learning_rate)
    scaler = torch.amp.GradScaler("cuda")  # Initialize the GradScaler

    dataset = SA1Folder([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14], args.images)

    if args.num_images is not None:
        dataset, _ = random_split(dataset, [args.num_images, len(dataset) - args.num_images])

    loader = DataLoader(dataset, shuffle=True, batch_size=args.batch_size, num_workers=args.num_workers)

    checkpoint_path = os.path.join(args.output_dir, "checkpoint.pth")
    
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        # get image encoder only
        sd = checkpoint['model']
        nanosam21.load_state_dict(sd, strict=False)
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resuming from epoch {start_epoch}")
    else:
        start_epoch = 0

    for epoch in range(start_epoch, args.num_epochs):

        epoch_loss = 0.

        for idx, image in enumerate(tqdm.tqdm(iter(loader))):
            image = image.cuda()
            if len(image) != args.batch_size:
                continue
            image_cnn = F.interpolate(image, (args.student_size, args.student_size), mode="area")

            with torch.no_grad():
                with torch.amp.autocast("cuda", torch.bfloat16):  # Enable mixed precision
                    features = sam21.image_encoder(image)["backbone_fpn"]

            optimizer.zero_grad()

            with torch.amp.autocast("cuda",torch.bfloat16):  # Enable mixed precision
                output = nanosam21.image_encoder(image_cnn)["backbone_fpn"]
                loss = 0
                for i in range(len(features)):
                    loss += loss_function(output[i], features[i]) * loss_weight[i]

            scaler.scale(loss).backward()  # Scale the loss for backpropagation
            scaler.step(optimizer)  # Update the parameters
            scaler.update()  # Update the scaler for next iteration

            epoch_loss += float(loss)

            if idx % 1000 == 0:
                plt.figure(figsize=(10, 10))
                plt.subplot(121)
                plt.imshow(features[-1][0,0].float().detach().cpu())
                plt.subplot(122)
                plt.imshow(output[-1][0,0].float().detach().cpu())
                plt.savefig(os.path.join(args.output_dir, f"epoch_{epoch}_batch_{idx}.png"))
                plt.close()

        epoch_loss /= len(loader)
        print(f"{epoch} - {epoch_loss}")

        with open(os.path.join(args.output_dir, 'log.txt'), 'a') as f:
            f.write(f"{epoch} - {epoch_loss}\n")

        torch.save({
            "model": nanosam21.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch}, checkpoint_path)

        plt.figure(figsize=(10, 10))
        plt.subplot(121)
        plt.imshow(features[-1][0,0].float().detach().cpu())
        plt.subplot(122)
        plt.imshow(output[-1][0,0].float().detach().cpu())
        plt.savefig(os.path.join(args.output_dir, f"epoch_{epoch}_batch_{idx}.png"))
        plt.close()
