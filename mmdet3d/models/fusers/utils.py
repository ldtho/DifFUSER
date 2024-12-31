import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def replace_nan(tensor: torch.Tensor, value: float = 1e-3) -> torch.Tensor:
    """
    Replaces NaN values in a tensor with a specified value.
    """
    return torch.where(torch.isnan(tensor), torch.full_like(tensor, value), tensor)


def check_nan(x: torch.Tensor, name: str) -> None:
    """
    Checks if a tensor has NaNs; if yes, prints a warning with mean and std.
    """
    if torch.isnan(x).any():
        print(f"{name} has nan: {x.mean()}, {x.std()}")


def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
    """
    Cosine schedule as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


def extract(a: torch.Tensor, t: torch.Tensor, x_shape: torch.Size) -> torch.Tensor:
    """
    Extracts values from a 1-D tensor `a` at indices specified by `t`,
    then reshapes them to match the shape of x except for the batch dimension.
    """
    batch_size = t.shape[0]
    out = a.gather(-1, t.long())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))


class SinusoidalPositionEmbeddings(nn.Module):
    """
    Generates sinusoidal positional embeddings of a given dimension.
    """
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


