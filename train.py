import torch
import torch.optim as optim
import torch.nn as nn
import os
import tqdm
import wandb
from models import Generator, Discriminator
from keys import api_key_wandb

def train(args, dataloader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize wandb
    wandb.login(key=api_key_wandb, anonymous='allow')
    wandb.init(project="face_completion_gan", config=args)
    config = wandb.config

    # Initialize models
    generator = Generator().to(device)
    discriminator = Discriminator().to(device)

    # Optimizers
    g_optimizer = optim.Adam(generator.parameters(), lr=config.lr, betas=(0.5, 0.999))
    d_optimizer = optim.Adam(discriminator.parameters(), lr=config.lr, betas=(0.5, 0.999))

    # Loss function
    criterion = nn.BCEWithLogitsLoss()

    # Create directory for checkpoints
    os.makedirs(config.checkpoint_dir, exist_ok=True)

    # Training loop
    total_steps = len(dataloader)
    for epoch in range(config.num_epochs):
        generator.train()
        discriminator.train()

        progress_bar = tqdm.tqdm(enumerate(dataloader), total=total_steps, desc=f"Epoch {epoch+1}/{config.num_epochs}")
        for i, (masked_images, real_images) in progress_bar:
            masked_images, real_images = masked_images.to(device), real_images.to(device)
            batch_size = real_images.size(0)

                        # Train Discriminator
            d_optimizer.zero_grad()
            # Real images
            real_output = discriminator(real_images)
            real_output = real_output.view(-1, 1)  # Flatten output
            real_labels = torch.ones_like(real_output).to(device)
            d_loss_real = criterion(real_output, real_labels)

            # Fake images
            fake_images = generator(masked_images)
            fake_output = discriminator(fake_images.detach())
            fake_output = fake_output.view(-1, 1)  # Flatten output
            fake_labels = torch.zeros_like(fake_output).to(device)
            d_loss_fake = criterion(fake_output, fake_labels)

            d_loss = (d_loss_real + d_loss_fake) / 2
            d_loss.backward()
            d_optimizer.step()

            # Train Generator
            g_optimizer.zero_grad()
            fake_output = discriminator(fake_images)
            g_loss = criterion(fake_output, real_labels)
            g_loss.backward()
            g_optimizer.step()

            # Log losses to wandb
            wandb.log({
                'D Loss': d_loss.item(),
                'G Loss': g_loss.item(),
                'epoch': epoch + 1
            })

            # Update progress bar
            progress_bar.set_postfix({
                'D Loss': f"{d_loss.item():.4f}",
                'G Loss': f"{g_loss.item():.4f}"
            })

        # Save checkpoint after each epoch
        checkpoint = {
            'epoch': epoch,
            'generator_state_dict': generator.state_dict(),
            'discriminator_state_dict': discriminator.state_dict(),
            'g_optimizer_state_dict': g_optimizer.state_dict(),
            'd_optimizer_state_dict': d_optimizer.state_dict(),
        }
        checkpoint_path = os.path.join(config.checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pth')
        torch.save(checkpoint, checkpoint_path)

        wandb.save(checkpoint_path)

        # Log generated images every 10 epochs
        with torch.no_grad():
            sample_masked = masked_images[:32].to(device)
            sample_generated = generator(sample_masked)
            wandb.log({"Generated Images": [wandb.Image(img) for img in sample_generated.cpu()]})

    # Save the final trained models
    torch.save(generator.state_dict(), os.path.join(config.checkpoint_dir, 'generator_final.pth'))
    torch.save(discriminator.state_dict(), os.path.join(config.checkpoint_dir, 'discriminator_final.pth'))

    wandb.save(os.path.join(config.checkpoint_dir, 'generator_final.pth'))
    wandb.save(os.path.join(config.checkpoint_dir, 'discriminator_final.pth'))

    print("Training completed.")
    wandb.finish()
