import tqdm
import torch
import argparse
from nanosam2.datasets.image_folder import SA1Folder
import os
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from nanosam2.sam2.build_sam import build_sam2
import json
from nanosam2.sam2.sam2_image_predictor import SAM2ImagePredictor
from torchvision.datasets import CocoDetection
from eval_coco import predict_box, box_xywh_to_xyxy, iou
from compute_eval_coco_metric import filter_results_by_area


def evaluate_on_coco(model, coco_dataset, max_images = -1):
    model.eval()
    predictor = SAM2ImagePredictor(model)
    results = []

    for i in tqdm.tqdm(range(len(coco_dataset)), desc="Evaluating on COCO"):
        image, anns = coco_dataset[i]

        for j, ann in enumerate(anns):
            box = box_xywh_to_xyxy(ann['bbox'])
            mask_coco = coco_dataset.coco.annToMask(ann)
            mask_coco = (mask_coco > 0)
            mask_sam = predict_box(predictor, image, box, set_image=(j==0))

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
        if max_images != -1 and i > max_images:
            break

    model.train()
    return results

def compute_metrics(results):
    miou_all = sum(r['iou'] for r in results) / len(results)
    miou_small = sum(r['iou'] for r in filter_results_by_area(results, None, 32**2)) / len(filter_results_by_area(results, None, 32**2))
    miou_medium = sum(r['iou'] for r in filter_results_by_area(results, 32**2, 96**2)) / len(filter_results_by_area(results, 32**2, 96**2))
    miou_large = sum(r['iou'] for r in filter_results_by_area(results, 96**2, None)) / len(filter_results_by_area(results, 96**2, None))

    return {
        "miou_all": miou_all,
        "miou_small": miou_small,
        "miou_medium": miou_medium,
        "miou_large": miou_large
    }

def perform_eval(nanosam21, coco_dataset, output_dir, num_eval_images):
    print(f"Evaluating on COCO dataset after epoch {epoch}")
    coco_results = evaluate_on_coco(nanosam21, coco_dataset, num_eval_images)
    metrics = compute_metrics(coco_results)
    
    print("COCO Evaluation Results:")
    print(f"mIOU (all): {metrics['miou_all']:.4f}")
    print(f"mIOU (small): {metrics['miou_small']:.4f}")
    print(f"mIOU (medium): {metrics['miou_medium']:.4f}")
    print(f"mIOU (large): {metrics['miou_large']:.4f}")

    if num_eval_images == -1:
        # Save evaluation results
        with open(os.path.join(output_dir, f'coco_eval_epoch_{epoch}.json'), 'w') as f:
            json.dump(coco_results, f)

        with open(os.path.join(output_dir, 'coco_eval_log.txt'), 'a') as f:
            f.write(f"Epoch {epoch}:\n")
            f.write(f"mIOU (all): {metrics['miou_all']:.4f}\n")
            f.write(f"mIOU (small): {metrics['miou_small']:.4f}\n")
            f.write(f"mIOU (medium): {metrics['miou_medium']:.4f}\n")
            f.write(f"mIOU (large): {metrics['miou_large']:.4f}\n\n")

    return metrics

loss_weight = [1.0, 1/4, 1/8]

if __name__ == "__main__":
        
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", required=True, type=str, help="The path to images to use for distillation")
    parser.add_argument("--output_dir", type=str, help="The directory to store checkpoints and training visualizations.")
    parser.add_argument("--num_images", type=int, default=None, help="Limit the number of images per epoch.")

    parser.add_argument("--num_epochs", type=int, default=100, help="The number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=12, help="The batch size.")
    parser.add_argument("--num_workers", type=int, default=8, help="The number of data loader workers.")
    parser.add_argument("--learning_rate", type=float, default=3e-4, help='The learning rate.')
    parser.add_argument("--log_interval", type=int, default=100, help="The number of batches to log training progress.")

    parser.add_argument("--teacher_checkpoint", type=str, default="sam2_checkpoints/sam2.1_hiera_large.pt", 
                        help="The path to a checkpoint to resume training.")
    parser.add_argument("--sam2_config", type=str, default="sam2.1_hiera_l", help="The path to a checkpoint to resume training.")
    parser.add_argument("--nanosam2_config", type=str, default="nanosam2.1_casvit", help="The path to a checkpoint to resume training.")
    parser.add_argument("--loss", type=str, default="huber", choices=["huber", "l1", "mse"], help="The loss function to use for distillation.")

    parser.add_argument("--qat", action="store_true", help="Use quantization aware training (QAT). Always train a FP32 first!")
    parser.add_argument("--fp32_checkpoint_for_qat", default=None, help="The path to a nanosam FP32 checkpoint to use for QAT.")
    
    parser.add_argument("--coco_root", type=str, default="/path/to/coco/val2017")
    parser.add_argument("--coco_ann", type=str, default="/path/to/coco/annotations/instances_val2017.json")
    parser.add_argument("--eval_interval", type=int, default=1, help="Number of epochs between evaluations")

    args = parser.parse_args()

    if args.fp32_checkpoint_for_qat is None and args.qat:
        raise ValueError("You must provide a FP32 checkpoint for QAT!")
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    sam21 = build_sam2(args.sam2_config, ckpt_path=args.teacher_checkpoint, 
            device="cuda", mode="eval", apply_postprocessing=False, load_image_encoder=True)
    nanosam21 = build_sam2(args.nanosam2_config, ckpt_path=args.teacher_checkpoint, 
            device="cuda", mode="eval", apply_postprocessing=False, load_image_encoder=False)

    if nanosam21.image_size != sam21.image_size:
        raise ValueError(f"Image size of student and teacher must be the same, got {nanosam21.image_size} and {sam21.image_size}")

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

    dataset = SA1Folder([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14], args.images, resolution=sam21.image_size)

    if args.num_images is not None:
        dataset, _ = random_split(dataset, [args.num_images, len(dataset) - args.num_images])

    loader = DataLoader(dataset, shuffle=True, batch_size=args.batch_size, num_workers=args.num_workers)

    checkpoint_path = os.path.join(args.output_dir, "checkpoint.pth")
    start_epoch = 0
    best_miou_all = 0.0

    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        nanosam21.load_state_dict(checkpoint['model'], strict=False)
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch'] + 1
        best_miou_all = checkpoint.get('best_miou_all', 0.0)
        print(f"Resuming from epoch {start_epoch}, best mIOU: {best_miou_all:.4f}")
    elif args.fp32_checkpoint_for_qat is not None and args.qat:
        print(f"Loading FP32 checkpoint for QAT from {args.fp32_checkpoint_for_qat}")
        checkpoint = torch.load(args.fp32_checkpoint_for_qat)
        nanosam21.load_state_dict(checkpoint['model'], strict=False)
    else:
        print("Starting training from scratch!")
 
    if args.qat:
        print("Enabling QAT for the image encoder")
        nanosam21.eval()
        nanosam21.image_encoder.qconfig = torch.ao.quantization.get_default_qat_qconfig('x86')
        nanosam21.image_encoder.train()
        torch.ao.quantization.prepare_qat(nanosam21.image_encoder, inplace=True)

    # Load COCO dataset for evaluation
    coco_dataset = CocoDetection(root=args.coco_root, annFile=args.coco_ann)

    for epoch in range(start_epoch, args.num_epochs):

        epoch_loss = 0.

        for idx, image in enumerate(tqdm.tqdm(iter(loader))):
            image = image.cuda()
            if len(image) != args.batch_size:
                continue
            image_cnn = F.interpolate(image, (sam21.image_size, sam21.image_size), mode="area")

            with torch.no_grad():
                with torch.amp.autocast("cuda", torch.bfloat16):
                    features = sam21.image_encoder(image)["backbone_fpn"]

            optimizer.zero_grad()

            with torch.amp.autocast("cuda", torch.bfloat16 if not args.qat else torch.float32):
                output = nanosam21.image_encoder(image_cnn)["backbone_fpn"]
                loss = 0
                for i in range(len(features)):
                    loss += loss_function(output[i], features[i]) * loss_weight[i]

            scaler.scale(loss).backward()  # Scale the loss for backpropagation
            scaler.step(optimizer)  # Update the parameters
            scaler.update()  # Update the scaler for next iteration

            epoch_loss += float(loss)

            if idx % args.log_interval == 0:
                plt.figure(figsize=(10, 10))
                plt.subplot(121)
                plt.imshow(features[-1][0,0].float().detach().cpu())
                plt.subplot(122)
                plt.imshow(output[-1][0,0].float().detach().cpu())
                plt.savefig(os.path.join(args.output_dir, f"epoch_{epoch}_batch_{idx}.png"))
                plt.close()

                perform_eval(nanosam21, coco_dataset, args.output_dir, num_eval_images=200)

        if epoch > 1:
            print("Disabling QAT observer")
            nanosam21.image_encoder.apply(torch.ao.quantization.disable_observer)

        if epoch == 1:
            print("Freezing BN stats")
            # Freeze batch norm mean and variance estimates
            nanosam21.image_encoder.apply(torch.nn.intrinsic.qat.freeze_bn_stats)

        epoch_loss /= len(loader)
        print(f"{epoch} - {epoch_loss}")

        with open(os.path.join(args.output_dir, 'log.txt'), 'a') as f:
            f.write(f"{epoch} - {epoch_loss}\n")

        metrics = perform_eval(nanosam21, coco_dataset, args.output_dir, -1)
        if metrics['miou_all'] > best_miou_all:
            print(f"New best mIOU (all): {best_miou_all:.4f}. Checkpoint saved.")
            best_miou_all = metrics['miou_all']
            torch.save({
                "model": nanosam21.state_dict(),
                "optimizer": optimizer.state_dict(),
                "best_miou_all": best_miou_all,
                "epoch": epoch}, checkpoint_path)

        plt.figure(figsize=(10, 10))
        plt.subplot(121)
        plt.imshow(features[-1][0,0].float().detach().cpu())
        plt.subplot(122)
        plt.imshow(output[-1][0,0].float().detach().cpu())
        plt.savefig(os.path.join(args.output_dir, f"epoch_{epoch}_batch_{idx}.png"))
        plt.close()
