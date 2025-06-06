import torchvision.utils as vutils
import matplotlib.pyplot as plt
from torchvision.transforms.functional import to_pil_image
import torch
import os 

def visualize_grouped_reconstructions(original, reconstructed, epoch, args):
    """Visualize RGB, NIR, and SWIR band reconstructions side-by-side."""
    imgs = []
    for i in range(original.shape[0]):  # for each of the 4 images
        sample = []

        # RGB: 3 channels
        rgb_orig = to_pil_image(original[i, [0, 1, 2]].cpu().clamp(0, 1))
        rgb_recon = to_pil_image(reconstructed[i, [0, 1, 2]].cpu().clamp(0, 1))
        sample.append(torch.cat([original[i, [0, 1, 2]], reconstructed[i, [0, 1, 2]]], dim=-1))

        # NIR: 3 channels
        nir_orig = to_pil_image(original[i, [3, 4, 5]].cpu().clamp(0, 1))
        nir_recon = to_pil_image(reconstructed[i, [3, 4, 5]].cpu().clamp(0, 1))
        sample.append(torch.cat([original[i, [3, 4, 5]], reconstructed[i, [3, 4, 5]]], dim=-1))

        # SWIR: single channel, display as grayscale
        swir_orig = original[i, 8].unsqueeze(0).cpu().clamp(0, 1)
        swir_recon = reconstructed[i, 8].unsqueeze(0).cpu().clamp(0, 1)
        sample.append(torch.cat([swir_orig, swir_recon], dim=-1))

        imgs.append(torch.cat(sample, dim=1))  # concatenate vertically (RGB/NIR/SWIR)

    grid = vutils.make_grid(imgs, nrow=1, normalize=True)


    if args.output_dir:
        vutils.save_image(grid, os.path.join(args.output_dir, f"recon_epoch_{epoch}.png"))
