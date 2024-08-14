import matplotlib.pyplot as plt
from dataloader import get_data_loader

def visualize_batch(data_loader):
    batch = next(iter(data_loader))
    skewed_left, skewed_right, original = batch

    fig, axes = plt.subplots(3, len(original), figsize=(15, 10))
    
    for i in range(len(original)):

        axes[0, i].imshow(original[i].permute(1, 2, 0) * 0.5 + 0.5)
        axes[0, i].axis('off')
        axes[0, i].set_title('Original')

        # Display the skewed left image (+X)
        axes[1, i].imshow(skewed_left[i].permute(1, 2, 0) * 0.5 + 0.5)
        axes[1, i].axis('off')
        axes[1, i].set_title('Left (+X)')

        # Display the skewed right image (-X)
        axes[2, i].imshow(skewed_right[i].permute(1, 2, 0) * 0.5 + 0.5)
        axes[2, i].axis('off')
        axes[2, i].set_title('Right (-X)')

    plt.tight_layout()
    plt.show()

# Example usage
data_dir = "/home/as-aravinthakshan/Desktop/Gan Project/"
data_loader = get_data_loader(data_dir, split='train', batch_size=8)
visualize_batch(data_loader)
