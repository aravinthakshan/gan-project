import argparse
from dataloader import get_data_loader
from train import train
from inference import inference
import torch

def parse_args():
    parser = argparse.ArgumentParser(description="Face Completion GAN")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "inference"], help="Train or inference mode")
    parser.add_argument("--data_dir", type=str, default="./data", help="Directory for the dataset")
    parser.add_argument("--split", type=str, default="train", choices=["train", "valid", "test"], help="Dataset split to use")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=0.0002, help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=200, help="Number of training epochs")
    parser.add_argument("--num_workers", type=int, default=2, help="Number of data loading workers")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--mask_side", type=str, default="random", choices=["random", "left", "right"], help="Side of the image to mask")
    parser.add_argument("--visualize", action="store_true", help="Visualize sample images during training")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume training from")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Get data loader
    dataloader = get_data_loader(
        args.data_dir, 
        batch_size=args.batch_size, 
        split=args.split, 
        num_workers=args.num_workers,
        mask_side=args.mask_side
    )

    if args.mode == "train":
        train(args, dataloader)
    elif args.mode == "inference":
        inference(args, device)
    else:
        raise ValueError("Invalid mode selected. Choose either 'train' or 'inference'.")

if __name__ == "__main__":
    main()