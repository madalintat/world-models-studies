"""ConvVAE over 64x64 RGB frames with a 32-dim latent.

This is the same model as the stage 0 compression VAE, redefined here so the
stage stays self contained. Architecture follows Ha and Schmidhuber 2018
(worldmodels.github.io): four stride-2 convs down to a 1024-dim feature,
then a decoder that starts from a 1x1x1024 tensor.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvVAE(nn.Module):
    def __init__(self, z_dim: int = 32):
        super().__init__()
        self.z_dim = z_dim
        self.enc = nn.Sequential(
            nn.Conv2d(3, 32, 4, stride=2), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2), nn.ReLU(),
            nn.Conv2d(128, 256, 4, stride=2), nn.ReLU(),
        )
        self.fc_mu = nn.Linear(1024, z_dim)
        self.fc_logvar = nn.Linear(1024, z_dim)
        self.fc_dec = nn.Linear(z_dim, 1024)
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(1024, 128, 5, stride=2), nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 5, stride=2), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 6, stride=2), nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 6, stride=2), nn.Sigmoid(),
        )

    def encode(self, x: torch.Tensor):
        h = self.enc(x).flatten(1)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor):
        std = (0.5 * logvar).exp()
        return mu + std * torch.randn_like(std)

    def decode(self, z: torch.Tensor):
        h = self.fc_dec(z).view(-1, 1024, 1, 1)
        return self.dec(h)

    def forward(self, x: torch.Tensor):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


def vae_loss(recon, x, mu, logvar, kl_tolerance: float = 0.5):
    """Sum-of-squares reconstruction plus KL with Ha's free-nats floor.

    The floor (kl_tolerance * z_dim) stops the optimizer from collapsing the
    posterior once the KL is already small, which would waste latent capacity.
    """
    recon_loss = F.mse_loss(recon, x, reduction="none").sum(dim=(1, 2, 3)).mean()
    kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=1)
    kl = torch.clamp(kl, min=kl_tolerance * mu.shape[1]).mean()
    return recon_loss + kl, recon_loss, kl
