import torch
from models import Generator
from dataloader import HalfMaskedFaceDataset
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision.utils as vutils
import os

def inference(args, device):
    # Load the trained generator
    generator = Generator().to(device)
    generator.load_state_dict(torch.load(args.generator_path, map_location=device))
    generator.eval()

    # Define the transform
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Create the dataset and dataloader
    sample_dataset = HalfMaskedFaceDataset(root=args.data_dir, transform=transform, mask_side=args.mask_side)
    sample_loader = DataLoader(sample_dataset, batch_size=args.batch_size, shuffle=True)

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Generate and save images
    for i, (masked_images, real_images) in enumerate(sample_loader):
        masked_images = masked_images.to(device)
        real_images = real_images.to(device)

        with torch.no_grad():
            generated_images = generator(masked_images)

        # Denormalize the images
        def denorm(x):
            return (x + 1) / 2

        # Concatenate masked, generated, and real images side by side
        comparison = torch.cat([denorm(masked_images), denorm(generated_images), denorm(real_images)], dim=3)

        # Save the comparison image
        vutils.save_image(comparison, os.path.join(args.output_dir, f'comparison_{i}.png'), nrow=1, normalize=False)

        if i >= args.num_samples - 1:
            break

    print(f"Generated images saved in {args.output_dir}")

# You would call this function from your main.py like this:
# inference(args, device)