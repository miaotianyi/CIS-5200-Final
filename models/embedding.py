import math
import torch
from torch import nn


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        """
        Sinusoidal position embedding as in diffusion and transformer

        The implementation is taken from
        https://huggingface.co/blog/annotated-diffusion

        This module doesn't have learnable parameters.

        The input is a tensor of shape [N, 1];
        The output is a tensor of shape [N, dim].

        Parameters
        ----------
        dim : int
            The number of output channels.
        """
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor):
        device = x.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = x[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

