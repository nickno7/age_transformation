import torch
from torchvision.utils import save_image
from torch.utils.data import Dataset
import os
from tqdm import tqdm


KAGGLE_URL = "nickno7/latent-codes-faces"
DATASET_PATH = "./dataset"

# dataset for generated images and their corresponding latents
class LatentDataset(Dataset):
    """Custom Dataset for loading generated images and their corresponding latent vectors."""
    def __init__(self, images, latents, transform=None):
        self.images = images  # List of images (PIL or numpy arrays)
        self.latents = latents  # Corresponding latent vectors
        self.transform = transform

    def __len__(self):
        """Returns the total number of samples in the dataset."""
        return len(self.images)

    def __getitem__(self, idx):
        """Fetches a sample (image, latent) from the dataset."""
        image = self.images[idx]
        latent = self.latents[idx]
        
        # Ensure latent is in the correct shape [1, 18, 512]
        if latent.dim() == 4:  # If shape is [1, 1, 324, 512]
            latent = latent.squeeze(0)  # Remove the extra dimension
        if latent.dim() == 3 and latent.shape[1] == 324:  # If shape is [1, 324, 512]
            latent = latent[:, :18, :]  # Truncate to [1, 18, 512]
        
        if self.transform:
            image = self.transform(image)
        
        return image, latent
    

def generate_dataset(device, g_mapping, g_synthesis, latent_dim=512, num_samples=10000, save_path=DATASET_PATH):
    """Generates a dataset of images and latents and saves them to disk."""
    images_path = os.path.join(save_path, "images")
    latents_path = os.path.join(save_path, "latents")

    os.makedirs(save_path, exist_ok=True)
    os.makedirs(images_path, exist_ok=True)
    os.makedirs(latents_path, exist_ok=True)

    # Find the highest existing index in the images and latents directories
    existing_images = [f for f in os.listdir(images_path) if f.endswith('.png')]
    existing_latents = [f for f in os.listdir(latents_path) if f.endswith('.pt')]

    if existing_images and existing_latents:
        # Ensure that the number of images and latents match
        assert len(existing_images) == len(existing_latents), "Mismatch between number of images and latents"
        
        # Find the highest index
        highest_index = max([int(f.split('.')[0]) for f in existing_images])
        start_index = highest_index + 1
    else:
        start_index = 0

    # Calculate the number of new samples to generate
    remaining_samples = num_samples - start_index

    if remaining_samples <= 0:
        print(f"Dataset already contains {start_index} samples, which is more than or equal to the requested {num_samples}.")
        return

    i = start_index
    for i in tqdm(range(start_index, start_index + remaining_samples), desc='Generating dataset...'):

        # Generate a random z vector
        z_sample = torch.randn(1, latent_dim).to(device) # [1, 512]
        # Pass through the mapping network to get the w vector
        latent = g_mapping(z_sample)

        # check to prevent shape errors
        if latent.shape[1] != 18:
            print(f"Incorrect latent shape {latent.shape} at sample {i}, skipping...")
            continue

        # Generate image
        with torch.no_grad():
            image = g_synthesis(latent)
        image = (image + 1.0) / 2.0  # Normalize to [0, 1]
        
        # Save image
        save_image(image.clamp(0, 1).cpu(), os.path.join(images_path, f"{i:05d}.png"))
        
        # Save latent
        torch.save(latent.cpu(), os.path.join(latents_path, f"{i:05d}.pt"))

    print("Dataset generation complete!")