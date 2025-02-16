import os
import torch
import torch.nn as nn
from torchvision.models import resnet50
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from tqdm import tqdm
from torch.utils.data import random_split
from src.dataset import LatentDataset, generate_dataset
import matplotlib.pyplot as plt


CHECKPOINT_FILE = "models/encoder_checkpoint.pt"
DATASET_PATH = "./dataset"

class Encoder(nn.Module):
    """Encoder network that uses ResNet50 as a backbone and learns to map images to latents."""
    def __init__(self, latent_dim=512, w_plus_layers=18):
        super(Encoder, self).__init__()
        # Load ResNet50 as a feature extractor
        self.resnet = resnet50(pretrained=True)
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-2])
        # Convolutional layer to adjust channels
        self.conv = nn.Conv2d(2048, latent_dim, kernel_size=1, stride=1)

        # Fully connected layers to map features to latent space
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(latent_dim * 7 * 7, 1024)
        self.fc2 = nn.Linear(1024, latent_dim * w_plus_layers)

    def forward(self, x):
        x = self.resnet(x)
        x = self.conv(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = x.view(-1, 18, 512)
        return x


def save_checkpoint(state, filename=CHECKPOINT_FILE):
    """Saves the current state of the model and optimizer to a checkpoint file."""
    torch.save(state, filename)
    print(f"Checkpoint saved to {filename}")


def load_checkpoint(device, model, optimizer, filename=CHECKPOINT_FILE):
    """Loads a checkpoint if available and resumes training from the last checkpoint."""
    if os.path.exists(filename):
        print(f"Loading checkpoint from {filename}")
        checkpoint = torch.load(filename, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        loss_history = checkpoint.get('loss_history', [])
        print(f"Resuming training from epoch {start_epoch}")
        return start_epoch, loss_history
    else:
        print("No checkpoint found, starting training from scratch.")
        return 0, []


def check_and_generate_dataset(device, dataset_path=DATASET_PATH, g_mapping=None, g_synthesis=None, num_samples=10000):
    """Check if a dataset exists and generate if no dataset is available"""
    images_path = os.path.join(dataset_path, "images")
    latents_path = os.path.join(dataset_path, "latents")

    # If either images or latents folder is empty, regenerate the dataset
    if not os.path.exists(images_path) or len(os.listdir(images_path)) == 0 or len(os.listdir(latents_path)) == 0:
        print("Dataset not found or is empty. Generating dataset...")
        generate_dataset(device, g_mapping, g_synthesis, num_samples=num_samples, save_path=dataset_path)
    else:
        print("Dataset found, proceeding with training.")


def logcosh_loss(original, generated):
    """Compute the log-cosh loss between original and generated latents."""
    loss = original - generated
    return torch.mean(torch.log(torch.cosh(loss + 1e-12)))


def evaluate(encoder, test_loader, device):
    """Evaluates the encoder model on the test dataset."""
    encoder.eval()  # Set the model to evaluation mode
    test_loss = 0.0

    with torch.no_grad():  # Disable gradient computation
        for imgs, true_latents in test_loader:
            imgs, true_latents = imgs.to(device), true_latents.to(device)

            # Forward pass
            predicted_latents = encoder(imgs)

            # Compute the loss
            loss = logcosh_loss(true_latents, predicted_latents)
            test_loss += loss.item()

    average_test_loss = test_loss / len(test_loader)
    return average_test_loss


def train(device, g_mapping, g_synthesis, dataset_dir=DATASET_PATH):
    """Training the Encoder model."""
    # Check if dataset exists, if not, generate it
    check_and_generate_dataset(device, DATASET_PATH, g_mapping=g_mapping, g_synthesis=g_synthesis, num_samples=10000)
    
    # Data transformation for images
    transform = Compose([
        Resize((224, 224)),
        ToTensor(),                       
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    dataset = LatentDataset(
        image_dir=os.path.join(dataset_dir, "images"), 
        latent_dir=os.path.join(dataset_dir, "latents"), 
        transform=transform
    )

    dataset_size = len(dataset)
    train_size = int(0.8 * dataset_size)  # 80% for training
    test_size = dataset_size - train_size  # 20% for testing
    
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    
    # Create DataLoaders for training and testing
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)
    
    # Initialize the encoder model
    encoder = Encoder().to(device)
    
    # Initialize optimizer
    optimizer = optim.Adam(encoder.parameters(), lr=3e-4, weight_decay=1e-5)
    
    
    # Check if there's an existing checkpoint
    start_epoch, loss_history = load_checkpoint(device, encoder, optimizer, filename=CHECKPOINT_FILE)

    train_losses = []
    test_losses = []
    
    epochs = 50
    for epoch in range(start_epoch, epochs):
        encoder.train()
        epoch_loss = 0
        
        with tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}", unit="batch") as pbar:
            for imgs, true_latents in pbar:
                imgs, true_latents = imgs.to(device), true_latents.to(device)    
    
                # Forward pass
                predicted_latents = encoder(imgs)

                # Compute the loss between the generated latent and the real latent
                loss = logcosh_loss(true_latents, predicted_latents) 
                epoch_loss += loss.item()
    
                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                pbar.set_postfix(loss=loss.item())

                del imgs, true_latents, predicted_latents, loss
                
        average_epoch_loss = epoch_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {average_epoch_loss:.4f}")
        loss_history.append(average_epoch_loss)

        # Evaluate on the test set
        train_losses.append(average_epoch_loss)
        test_loss = evaluate(encoder, test_loader, device)
        test_losses.append(test_loss)
        print(f"Epoch [{epoch+1}/{epochs}], Test Loss: {test_loss:.4f}")

        # Save checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': encoder.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': average_epoch_loss,
            'test_loss': test_loss
        }
        save_checkpoint(checkpoint, filename=CHECKPOINT_FILE)

    # Plot training and test loss
    plt.plot(train_losses, label="Training Loss")
    plt.plot(test_losses, label="Test Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()
    
    print("Training complete!")