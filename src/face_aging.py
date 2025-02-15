import numpy as np
import torch
import imageio
from PIL import Image

def load_age_boundary(device):
    """Load the age boundary file"""
    age_boundary = np.load("/content/drive/MyDrive/stylegan_ffhq_age_w_boundary.npy")
    age_boundary = torch.tensor(age_boundary, dtype=torch.float32).unsqueeze(0)
    age_boundary = age_boundary.to(device)
    return age_boundary

def manipulate_latent(w, boundary, alpha):
    """Calculate the new latent vector with applied age factor"""
    return w + alpha * boundary

def create_gif(latent, aged_w, g_synthesis, display_gif=True):
    # Number of interpolation steps for a smoother transition
    num_steps = 75

    # Generate interpolated images
    interpolated_images = []
    with torch.no_grad():
        for a in np.linspace(0, 1, num_steps):
            z = ((1 - a) * latent) + (a * aged_w)
            result = g_synthesis(z)
            result = (result.clamp(-1, 1) + 1) / 2.0  # Normalize
            result = result.cpu()
            interpolated_images.append(result)

    # Prepare the images for the GIF
    images_for_gif = []
    for img_tensor in interpolated_images:
        # Convert tensor to numpy array
        img = img_tensor.squeeze(0).permute(1, 2, 0).detach().numpy()
        img = (img * 255).astype(np.uint8)  # Convert to uint8 format for saving
        images_for_gif.append(img)

    # Create a GIF
    output_gif_path = "outputs/age_transition.gif"
    imageio.mimsave(output_gif_path, images_for_gif, loop=0, fps=10)  # Adjust fps for speed

    print(f"GIF saved as {output_gif_path}")

    # display the GIF
    if display_gif:
        img = Image.open(output_gif_path)
        img.show()  # Opens the GIF in a new window

def age_progression(device, latent, alpha, g_synthesis, display_gif=True):
    # load pre-trained age boundary
    age_boundary = load_age_boundary(device)

    # calculate aged latent
    aged_w = manipulate_latent(latent, age_boundary, alpha)

    # create and display resulting GIF
    create_gif(latent, aged_w, g_synthesis, display_gif)