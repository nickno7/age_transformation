import torch
import argparse
import os
from src.dataset import generate_dataset, download_and_extract_dataset
from src.latent_encoder import Encoder, train
from src.latent_optimization import optimize_latent
from src.face_aging import age_progression
from src.stylegan_implementation import G_mapping, G_synthesis
from collections import OrderedDict
import kagglehub

CHECKPOINT_PATH = "models/encoder_checkpoint.pt"
DATASET_PATH = "./dataset"

def setup_stylegan(device):
    """Setup StyleGAN with pre-trained weights."""
    print("Setting up StyleGAN model...")
    g_all = torch.nn.Sequential(OrderedDict([
        ('g_mapping', G_mapping()),
        ('g_synthesis', G_synthesis())
    ]))

    g_mapping = g_all[0]
    g_synthesis = g_all[1]
    
    # Download and load pretrained weights
    path = kagglehub.dataset_download("songseungwon/ffhq-1024x1024-pretrained")
    g_all.load_state_dict(torch.load(os.path.join(path, 'karras2019stylegan-ffhq-1024x1024.for_g_all.pt')))
    g_all.eval().to(device)
    
    return g_mapping, g_synthesis

def load_checkpoint(encoder, checkpoint_path, device):
    """Loads model weights from a checkpoint if available."""
    print(f"Loading trained encoder from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    encoder.load_state_dict(checkpoint["model_state_dict"])
    encoder.eval()

def load_encoder(device, checkpoint_path, g_mapping, g_synthesis):
    """Loads the trained encoder model. If not available train the encoder"""
    encoder = Encoder().to(device)
    
    if os.path.exists(checkpoint_path):
        load_checkpoint(encoder, checkpoint_path, device)
    else:
        print("No trained encoder found. Starting training...")
        train(device, g_mapping, g_synthesis)
        load_checkpoint(encoder, checkpoint_path, device)
    
    return encoder

def main():
    parser = argparse.ArgumentParser(description="Age Progression with StyleGAN")
    parser.add_argument("--image", type=str, help="Path to input image for optimization")
    parser.add_argument("--random", action="store_true", help="Use a randomly generated StyleGAN face for age transformation")
    parser.add_argument("--generate", action="store_true", help="Generate dataset")

    parser.add_argument("--num_samples", type=int, default=10000, help="Number of samples to process")
    parser.add_argument("--save_path", type=str, default="./dataset", help="Path to save the dataset")
    parser.add_argument("--display_gif", type=bool, default=True, help="Whether to display the resulting GIF")
    parser.add_argument("--alpha", type=int, default=3.0, help="Aging Intensity")

    
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    g_mapping, g_synthesis = setup_stylegan(device)
    
    if args.generate:
        generate_dataset(device, g_mapping, g_synthesis, num_samples=args.num_samples, save_path=args.save_path)

    elif args.image:
        # Ensure dataset exists
        download_and_extract_dataset(save_path=args.save_path)
        print(f"Optimizing latent for image: {args.image}")
        encoder_path = kagglehub.model_download('nickno7/encoder/PyTorch/default/1')
        encoder = load_encoder(device, encoder_path)
        latent = optimize_latent(args.image, device, encoder, g_synthesis)
        age_progression(device, latent, args.alpha, g_synthesis, display_gif=args.display_gif)

    elif args.random:
        print("Generating and aging a random StyleGAN face...")
        z_sample = torch.randn(1, 512).to(device)  # Generate random latent vector
        latent = g_mapping(z_sample)
        age_progression(device, latent, args.alpha, g_synthesis, display_gif=args.display_gif)

    else:
        print("No operation specified. Use --generate, --image, or --random.")

if __name__ == "__main__":
    main()
