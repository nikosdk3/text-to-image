import torch
from torch import nn
import torch.nn.functional as F


def exists(val):
    return val is not None


def cosine_beta_schedule(timesteps, s=0.008):
    """
    https://www.zainnasir.com/blog/cosine-beta-schedule-for-denoising-diffusion-models/
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:]) / (alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


class BaseGaussianDiffusion(nn.Module):
    def __init__(self, *, beta_schedule, timesteps, loss_type):
        super().__init__()

        if beta_schedule == "cosine":
            betas = co


class TextToImage(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x
