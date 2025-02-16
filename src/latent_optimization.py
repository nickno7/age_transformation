import torch
import torch.optim as optim
import torch.nn as nn
from PIL import Image
from torchvision import models
from torchvision.utils import save_image
import torchvision.transforms as transforms


class ResNet_perceptual(nn.Module):
    """ResNet-based perceptual model."""
    def __init__(self, requires_grad=False):
        super(ResNet_perceptual, self).__init__()
        resnet = models.resnet50(pretrained=True)
        self.slice1 = nn.Sequential(*list(resnet.children())[:3])
        self.slice2 = nn.Sequential(*list(resnet.children())[3:5])
        self.slice3 = nn.Sequential(*list(resnet.children())[5:6])
        self.slice4 = nn.Sequential(*list(resnet.children())[6:7])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h = self.slice1(X)
        h_relu1 = h
        h = self.slice2(h)
        h_relu2 = h
        h = self.slice3(h)
        h_relu3 = h
        h = self.slice4(h)
        h_relu4 = h
        return h_relu1, h_relu2, h_relu3, h_relu4


def load_image(path, device):
    """Loads and transforms an image for processing."""
    with open(path, "rb") as f:
        image = Image.open(f).convert("RGB")

    transform_encoder = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Resize((1024, 1024))])

    encoder_image = transform_encoder(image).unsqueeze(0).to(device)
    image = transform(image).unsqueeze(0).to(device)

    return image, encoder_image


def loss_function(syn_img, img, img_p, mse_loss, upsample, perceptual):
    """Computes MSE and perceptual loss."""
    syn_img_p = upsample(syn_img)
    syn_feats = perceptual(syn_img_p)
    real_feats = perceptual(img_p)

    mse = mse_loss(syn_img, img)
    per_loss = sum(mse_loss(syn, real) for syn, real in zip(syn_feats,
                                                            real_feats))

    return mse, per_loss


def optimize_latent(path, device, encoder, g_synthesis):
    """Optimizes latent representation of an input image."""
    # load an transform image
    image, encoder_image = load_image(path, device)

    upsample = torch.nn.Upsample(scale_factor=224/1024, mode='bilinear')
    img_p = upsample(image.clone())
    perceptual = ResNet_perceptual().to(device)

    # MSE loss object
    mse_loss = nn.MSELoss()
    # send image through encoder for first approximation
    with torch.no_grad():
        latent = encoder(encoder_image)
    # Optimize latent code in each backward step
    optimizer = optim.Adam({latent}, lr=0.01, betas=(0.9, 0.999), eps=1e-8)

    loss_ = []

    print("Starting Optimization loop. This will take a moment...")
    # latent optimization loop
    for i in range(2000):

        optimizer.zero_grad()
        syn_img = g_synthesis(latent)
        syn_img = (syn_img + 1.0)/2.0

        # calculate losses
        mse, per_loss = loss_function(syn_img, image, img_p,
                                      mse_loss, upsample, perceptual)

        loss = 0.6 * per_loss + 0.4 * mse
        loss.backward()
        optimizer.step()

        loss_np = loss.detach().cpu().numpy()
        loss_mse = mse.detach().cpu().numpy()
        loss_per = per_loss.detach().cpu().numpy()
        loss_.append(loss_np)

        # print loss and save image every 500th iteration
        if (i + 1) % 50 == 0:
            print(f"Iteration{i+1}: loss -- {loss_np},  mse_loss -- {loss_mse},  percep_loss -- {loss_per}")
            save_image(syn_img.clamp(0, 1), f"outputs/face_{i+1}_iterations.png")

    return latent
