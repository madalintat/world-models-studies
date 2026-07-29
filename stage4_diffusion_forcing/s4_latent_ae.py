"""Small deterministic conv autoencoder for 64x64 frames.

This mirrors stage 0's ConvVAE but drops the variational part: no mu/logvar,
no KL term, just an encoder to an 8x8 spatial grid with `latent_dim` channels
and a decoder back to pixels. The dynamics model treats the grid as 64 tokens
of dim `latent_dim`.
"""

import torch
import torch.nn as nn
from einops import rearrange


class LatentAE(nn.Module):
    def __init__(self, latent_dim: int = 8, base: int = 32):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            nn.Conv2d(3, base, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(base, base * 2, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(base * 2, base * 4, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(base * 4, latent_dim, 3, 1, 1),
            # tanh keeps latents bounded so their scale stays close to the
            # unit-variance noise used by flow matching
            nn.Tanh(),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(latent_dim, base * 4, 3, 1, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(base * 4, base * 2, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(base * 2, base, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(base, 3, 4, 2, 1),
            nn.Sigmoid(),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """(B, 3, 64, 64) in [0, 1] -> (B, latent_dim, 8, 8)"""
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """(B, latent_dim, 8, 8) -> (B, 3, 64, 64) in [0, 1]"""
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))

    @staticmethod
    def to_tokens(z: torch.Tensor) -> torch.Tensor:
        """(B, D, 8, 8) -> (B, 64, D)"""
        return rearrange(z, "b d h w -> b (h w) d")

    @staticmethod
    def from_tokens(tokens: torch.Tensor, grid: int = 8) -> torch.Tensor:
        """(B, 64, D) -> (B, D, 8, 8)"""
        return rearrange(tokens, "b (h w) d -> b d h w", h=grid, w=grid)
