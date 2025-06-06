import torchvision.utils as vutils
import matplotlib.pyplot as plt
from torchvision.transforms.functional import to_pil_image
import torch
import os 

import torchvision.utils as vutils
from torchvision.transforms.functional import to_pil_image
import torch

def denormalize(tensor, mean, std):
    """
    Denormalize a tensor with shape [C, H, W] using per-channel mean and std,
    then scale to 0-255.
    """
    mean = torch.tensor(mean).view(-1, 1, 1).to(tensor.device)
    std = torch.tensor(std).view(-1, 1, 1).to(tensor.device)
    return (tensor * std + mean) 

# Denormalization values
mean = [1370.19151926, 1184.3824625, 1120.77120066, 1136.26026392,
        1263.73947144, 1645.40315151, 1846.87040806, 1762.59530783,
        1972.62420416, 582.72633433, 14.77112979, 1732.16362238, 1247.91870117]

std = [633.15169573, 650.2842772, 712.12507725, 965.23119807,
       948.9819932, 1108.06650639, 1258.36394548, 1233.1492281,
       1364.38688993, 472.37967789, 14.3114637, 1310.36996126, 1087.6020813]


def visualize_grouped_reconstructions(original, reconstructed, epoch, args, log_wandb=True):
    imgs = []
    # for i in range(original.shape[0]):  # Loop over 4 images
    #     # RGB
    #     nnorm = (original[i, [0, 1, 2]].cpu(), mean[:3], std[:3]).clamp(0, 1)
    #     rgb_orig = denormalize(original[i, [0, 1, 2]].cpu(), mean[:3], std[:3]).clamp(0, 1)
    #     rgb_recon = denormalize(reconstructed[i, [0, 1, 2]].cpu(), mean[:3], std[:3]).clamp(0, 1)
    #     print("RGB original min/max:", rgb_orig.min(), rgb_orig.max())
    #     print("nnorm original min/max:", nnorm.min(), nnorm.max())


    #     to_pil_image(rgb_orig).save(os.path.join(args.output_dir, f"epoch{epoch}_img{i}_rgb_orig.png"))
    #     to_pil_image(rgb_recon).save(os.path.join(args.output_dir, f"epoch{epoch}_img{i}_rgb_recon.png"))

    #     # NIR
    #     nir_orig = denormalize(original[i, [3, 4, 5]].cpu(), mean[3:6], std[3:6]).clamp(0, 1)
    #     nir_recon = denormalize(reconstructed[i, [3, 4, 5]].cpu(), mean[3:6], std[3:6]).clamp(0, 1)

    #     to_pil_image(nir_orig).save(os.path.join(args.output_dir, f"epoch{epoch}_img{i}_nir_orig.png"))
    #     to_pil_image(nir_recon).save(os.path.join(args.output_dir, f"epoch{epoch}_img{i}_nir_recon.png"))

    #     # SWIR
    #     swir_orig = denormalize(original[i, 8].unsqueeze(0).cpu(), [mean[8]], [std[8]]).clamp(0, 1)
    #     swir_recon = denormalize(reconstructed[i, 8].unsqueeze(0).cpu(), [mean[8]], [std[8]]).clamp(0, 1)

    #     to_pil_image(swir_orig).save(os.path.join(args.output_dir, f"epoch{epoch}_img{i}_swir_orig.png"))
    #     to_pil_image(swir_recon).save(os.path.join(args.output_dir, f"epoch{epoch}_img{i}_swir_recon.png"))

    for i in range(original.shape[0]):  # Loop over 4 images
        panels = []

        # RGB: [0, 1, 2]
        # rgb_orig = original[i, [0, 1, 2]].cpu().clamp(0, 1)
        # rgb_recon = reconstructed[i, [0, 1, 2]].cpu().clamp(0, 1)
        # RGB (bands 0,1,2)
        rgb_orig = denormalize(original[i, [0, 1, 2]].cpu(), mean[:3], std[:3]).clamp(0, 1)
        rgb_recon = denormalize(reconstructed[i, [0, 1, 2]].cpu(), mean[:3], std[:3]).clamp(0, 1)

        panels.append(torch.cat([rgb_orig, rgb_recon], dim=-1))

        # NIR: [3, 4, 5]
        # nir_orig = original[i, [3, 4, 5]].cpu().clamp(0, 1)
        # nir_recon = reconstructed[i, [3, 4, 5]].cpu().clamp(0, 1)
        # NIR: bands [3, 4, 5]
        nir_orig = denormalize(original[i, [3, 4, 5]].cpu(), mean[3:6], std[3:6]).clamp(0, 1)
        nir_recon = denormalize(reconstructed[i, [3, 4, 5]].cpu(), mean[3:6], std[3:6]).clamp(0, 1)

        # SWIR: band 8
        swir_orig = denormalize(original[i, 8].unsqueeze(0).cpu(), [mean[8]], [std[8]]).clamp(0, 1)
        swir_recon = denormalize(reconstructed[i, 8].unsqueeze(0).cpu(), [mean[8]], [std[8]]).clamp(0, 1)

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
        stacked = torch.cat(panels, dim=1)
        imgs.append(stacked)

    # grid = vutils.make_grid(imgs, nrow=1, normalize=True)


    # if args.output_dir:
    #     vutils.save_image(grid, os.path.join(args.output_dir, f"recon_epoch_{epoch}.png"))
