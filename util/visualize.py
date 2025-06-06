import torchvision.utils as vutils
import matplotlib.pyplot as plt
from torchvision.transforms.functional import to_pil_image
import torch
import os 

import torchvision.utils as vutils
from torchvision.transforms.functional import to_pil_image

def visualize_grouped_reconstructions(original, reconstructed, epoch, args, log_wandb=True):
    imgs = []

    for i in range(original.shape[0]):  # Loop over 4 images
        panels = []

        # RGB: [0, 1, 2]
        rgb_orig = original[i, [0, 1, 2]].cpu().clamp(0, 1)
        rgb_recon = reconstructed[i, [0, 1, 2]].cpu().clamp(0, 1)
        panels.append(torch.cat([rgb_orig, rgb_recon], dim=-1))

        # NIR: [3, 4, 5]
        nir_orig = original[i, [3, 4, 5]].cpu().clamp(0, 1)
        nir_recon = reconstructed[i, [3, 4, 5]].cpu().clamp(0, 1)
        panels.append(torch.cat([nir_orig, nir_recon], dim=-1))

        # SWIR: [8] – single channel, grayscale
        swir_orig = original[i, 8].unsqueeze(0).cpu().clamp(0, 1)
        swir_recon = reconstructed[i, 8].unsqueeze(0).cpu().clamp(0, 1)
        panels.append(torch.cat([swir_orig, swir_recon], dim=-1))
        print(len(panels))
        print(panels[0].shape)
        print(panels[1].shape)
        print(panels[2].shape)
        rgb_grid = vutils.make_grid([rgb_orig, rgb_recon], nrow=2, normalize=True)
        vutils.save_image(rgb_grid, os.path.join(args.output_dir, f"epoch{epoch}_img{i}_rgb.png"))
        nir_grid = vutils.make_grid([nir_orig, nir_recon], nrow=2, normalize=True)
        vutils.save_image(nir_grid, os.path.join(args.output_dir, f"epoch{epoch}_img{i}_nir.png"))
        swir_grid = vutils.make_grid([swir_orig, swir_recon], nrow=2, normalize=True)
        vutils.save_image(swir_grid, os.path.join(args.output_dir, f"epoch{epoch}_img{i}_swir.png"))
        # Stack vertically: RGB, NIR, SWIR (all on CPU)
    #     stacked = torch.cat(panels, dim=1)
    #     imgs.append(stacked)

    # grid = vutils.make_grid(imgs, nrow=1, normalize=True)


    # if args.output_dir:
    #     vutils.save_image(grid, os.path.join(args.output_dir, f"recon_epoch_{epoch}.png"))
