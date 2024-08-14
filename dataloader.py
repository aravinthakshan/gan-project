import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os

class HalfMaskedFaceDataset(Dataset):
    def __init__(self, root, transform=None, mask_side='random'):
        self.root = root
        self.transform = transform
        self.image_paths = [os.path.join(root, fname) for fname in os.listdir(root) if fname.endswith(('.jpg', '.png'))]
        self.mask_side = mask_side

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert('RGB')
        
        if self.transform:
            img = self.transform(img)
        
        _, h, w = img.shape
        
        # Create a mask (0 for black, 1 for original image)
        mask = torch.ones_like(img)
        
        if self.mask_side == 'random':
            mask_left = torch.rand(1) > 0.5
        elif self.mask_side == 'left':
            mask_left = True
        elif self.mask_side == 'right':
            mask_left = False
        else:
            raise ValueError("mask_side must be 'random', 'left', or 'right'")
        
        if mask_left:
            mask[:, :, :w//2] = 0
        else:
            mask[:, :, w//2:] = 0
        
        # Apply the mask to the image
        masked_img = img * mask
        
        return masked_img, img

def get_data_loader(data_dir, batch_size=32, split='train', num_workers=4, mask_side='random'):
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    dataset = HalfMaskedFaceDataset(root=os.path.join(data_dir, split), transform=transform, mask_side=mask_side)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)