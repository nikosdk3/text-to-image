from matplotlib import axis
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


def linear_beta_schedule(timesteps):
    scale = timesteps / 1000
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)


def quadratic_beta_schedule(timesteps):
    scale = timesteps / 1000
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return (
        torch.linspace(beta_start**2, beta_end**2, timesteps, dtype=torch.float64) ** 2
    )


def sigmoid_beta_schedule(timesteps):
    scale = timesteps / 1000
    beta_start = timesteps * 0.0001
    beta_end = timesteps * 0.02
    betas = torch.linspace(-6, 6, timesteps, dtype=torch.float64)
    return torch.sigmoid(betas) * (beta_end - beta_start) + beta_start


class BaseGaussianDiffusion(nn.Module):
    def __init__(self, *, beta_schedule, timesteps, loss_type):
        super().__init__()

        if beta_schedule == "cosine":
            betas = cosine_beta_schedule(timesteps)
        elif beta_schedule == "linear":
            betas = linear_beta_schedule(timesteps)
        elif beta_schedule == "quadratic":
            betas = quadratic_beta_schedule(timesteps)
        elif beta_schedule == "sigmoid":
            betas = sigmoid_beta_schedule(timesteps)
        elif beta_schedule == "jsd":
            betas = 1.0 / torch.linspace(timesteps, 1, timesteps)
        else:
            raise NotImplementedError()

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)

        if loss_type == "l1":
            loss_fn = F.l1_loss
        elif loss_type == "l2":
            loss_fn = F.mse_loss
        else:
            raise NotImplementedError()

        self.loss_type = loss_type
        self.loss_fn = loss_fn



class TextToImage(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x
