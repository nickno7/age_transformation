import numpy as np
import torch
import imageio
import gdown
import os
import platform
import cv2
import time


def load_age_boundary(device):
    """Download and load the age boundary file from Google Drive using gdown"""

    # Correct Google Drive file ID
    file_id = "1-fYz2hSegMjkohBq26Fdx6eGkyGaKe1r"

    # URL for downloading from Google Drive
    url = f"https://drive.google.com/uc?id={file_id}"

    # Path where the file will be saved locally
    age_boundary_path = "./stylegan_ffhq_age_w_boundary.npy"

    # Download the file using gdown if not already downloaded
    if not os.path.exists(age_boundary_path):
        print("Downloading the age boundary file...")
        gdown.download(url, age_boundary_path, quiet=False)

    # Load the file and convert it to a PyTorch tensor
    age_boundary = np.load(age_boundary_path)
    age_boundary = torch.tensor(age_boundary, dtype=torch.float32).unsqueeze(0)
    age_boundary = age_boundary.to(device)

    return age_boundary


def manipulate_latent(w, boundary, alpha):
    """Calculate the new latent vector with applied age factor"""
    return w + alpha * boundary


def create_gif(latent, aged_w, g_synthesis, display_gif=True):
    """Interpolate images and create a GIF"""
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
        img = (img * 255).astype(np.uint8)
        images_for_gif.append(img)

    # get timestamp as unique id
    timestamp = int(time.time())

    # Ensure the output directories exist
    gif_directory = "outputs/gifs"
    video_directory = "outputs/videos"
    os.makedirs(gif_directory, exist_ok=True)
    os.makedirs(video_directory, exist_ok=True)

    # Create a GIF
    output_gif_path = f"{gif_directory}/age_transition_{timestamp}.gif"
    imageio.mimsave(output_gif_path, images_for_gif, loop=0, duration=10)  # Adjust fps for speed

    print(f"GIF saved as {output_gif_path}")

    # convert gif to mp4 video an display it
    if display_gif:
        # Convert GIF to video
        video_path = f"{video_directory}/age_transition_{timestamp}.mp4"
        frame_height, frame_width, _ = images_for_gif[0].shape
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(video_path, fourcc, 10,
                                       (frame_width, frame_height))

        for img in images_for_gif:
            # Convert from RGB to BGR for OpenCV
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            video_writer.write(img_bgr)

        video_writer.release()  # Finalize the video
        print(f"Video saved as {video_path}")

        # Display the video in default video player
        if platform.system() == "Darwin":  # macOS
            os.system(f"open {video_path}")
        elif platform.system() == "Windows":
            os.startfile(video_path)
        else:
            os.system(f"xdg-open {video_path}")  # For Linux or others


def age_progression(device, latent, alpha, g_synthesis, display_gif=True):
    """Full process for aging a face by manipulating the latent.
        GIF of the age transformation as Output."""
    # load pre-trained age boundary
    age_boundary = load_age_boundary(device)

    # calculate aged latent
    aged_w = manipulate_latent(latent, age_boundary, alpha)

    # create and display resulting GIF
    create_gif(latent, aged_w, g_synthesis, display_gif)
