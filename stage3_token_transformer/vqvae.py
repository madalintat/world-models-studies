"""VQ-VAE for 64x64 CarRacing frames.

Encoder maps a frame to an 8x8 grid of continuous vectors, the quantizer snaps
each vector to its nearest codebook entry, the decoder reconstructs pixels from
the snapped grid. The codebook is trained with EMA updates (no gradient), the
encoder gets gradients through the straight-through estimator plus a commitment
loss.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class VectorQuantizerEMA(nn.Module):
    """Nearest-neighbor quantizer with EMA codebook updates and dead-code reinit.

    enable_ema / enable_dead_reinit exist so the break-it labs can switch the
    stabilizers off and watch the codebook collapse.
    """

    def __init__(
        self,
        num_codes: int = 256,
        code_dim: int = 64,
        beta: float = 0.25,
        decay: float = 0.99,
        eps: float = 1e-5,
        dead_threshold: float = 0.2,
        enable_ema: bool = True,
        enable_dead_reinit: bool = True,
    ):
        super().__init__()
        self.num_codes = num_codes
        self.code_dim = code_dim
        self.beta = beta
        self.decay = decay
        self.eps = eps
        self.dead_threshold = dead_threshold
        self.enable_ema = enable_ema
        self.enable_dead_reinit = enable_dead_reinit

        codebook = torch.randn(num_codes, code_dim) * 0.1
        self.register_buffer("codebook", codebook)
        self.register_buffer("ema_cluster_size", torch.ones(num_codes))
        self.register_buffer("ema_embed_sum", codebook.clone())
        # Cumulative pick counts, for the usage histogram.
        self.register_buffer("usage", torch.zeros(num_codes))

    def forward(self, z_e: torch.Tensor):
        """z_e: (B, code_dim, H, W). Returns (z_q_st, indices, info)."""
        B, D, H, W = z_e.shape
        flat = rearrange(z_e, "b d h w -> (b h w) d")
        dist = (
            flat.pow(2).sum(1, keepdim=True)
            - 2.0 * flat @ self.codebook.t()
            + self.codebook.pow(2).sum(1)
        )
        indices = dist.argmin(dim=1)
        z_q = self.codebook[indices].view(B, H, W, D).permute(0, 3, 1, 2)

        commit = self.beta * F.mse_loss(z_e, z_q.detach())

        with torch.no_grad():
            counts = torch.bincount(indices, minlength=self.num_codes).float()
            self.usage += counts
            probs = counts / counts.sum()
            perplexity = torch.exp(-(probs * (probs + 1e-10).log()).sum())
            if self.training and self.enable_ema:
                self._ema_update(flat.detach(), indices, counts)
            if self.training and self.enable_dead_reinit:
                self._reinit_dead(flat.detach())

        # Straight-through: forward uses z_q, backward copies gradients to z_e.
        z_q_st = z_e + (z_q - z_e).detach()
        info = {"commit": commit, "perplexity": perplexity}
        return z_q_st, indices.view(B, H, W), info

    def _ema_update(self, flat, indices, counts):
        onehot = F.one_hot(indices, self.num_codes).to(flat.dtype)
        embed_sum = onehot.t() @ flat
        self.ema_cluster_size.mul_(self.decay).add_(counts, alpha=1 - self.decay)
        self.ema_embed_sum.mul_(self.decay).add_(embed_sum, alpha=1 - self.decay)
        n = self.ema_cluster_size.sum()
        # Laplace smoothing keeps rarely used codes from dividing by ~0.
        smoothed = (self.ema_cluster_size + self.eps) / (n + self.num_codes * self.eps) * n
        self.codebook.copy_(self.ema_embed_sum / smoothed.unsqueeze(1))

    def _reinit_dead(self, flat):
        dead = self.ema_cluster_size < self.dead_threshold
        n_dead = int(dead.sum())
        if n_dead == 0:
            return
        pick = torch.randint(0, flat.shape[0], (n_dead,), device=flat.device)
        src = flat[pick]
        self.codebook[dead] = src
        self.ema_embed_sum[dead] = src
        self.ema_cluster_size[dead] = 1.0

    def usage_histogram(self) -> torch.Tensor:
        h = self.usage.clone()
        if h.sum() > 0:
            h = h / h.sum()
        return h

    def active_codes(self) -> int:
        return int((self.usage > 0).sum())


class VQEncoder(nn.Module):
    def __init__(self, code_dim: int = 64, base: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, base, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(base, base * 2, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(base * 2, base * 2, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(base * 2, code_dim, 3, 1, 1),
        )

    def forward(self, x):
        return self.net(x)


class VQDecoder(nn.Module):
    def __init__(self, code_dim: int = 64, base: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(code_dim, base * 2, 3, 1, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(base * 2, base * 2, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(base * 2, base, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(base, 3, 4, 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, z):
        return self.net(z)


class VQVAE(nn.Module):
    """64x64x3 in [0,1] -> 8x8 grid of code indices -> 64x64x3 in [0,1]."""

    def __init__(self, num_codes: int = 256, code_dim: int = 64, base: int = 64, **quant_kwargs):
        super().__init__()
        self.encoder = VQEncoder(code_dim, base)
        self.quantizer = VectorQuantizerEMA(num_codes, code_dim, **quant_kwargs)
        self.decoder = VQDecoder(code_dim, base)
        self.grid_hw = 8
        self.tokens_per_frame = self.grid_hw * self.grid_hw

    def forward(self, x):
        z_e = self.encoder(x)
        z_q, indices, info = self.quantizer(z_e)
        recon = self.decoder(z_q)
        return recon, indices, info

    def loss(self, x):
        recon, indices, info = self.forward(x)
        recon_loss = F.mse_loss(recon, x)
        total = recon_loss + info["commit"]
        return total, {
            "recon": recon_loss.detach(),
            "commit": info["commit"].detach(),
            "perplexity": info["perplexity"],
        }

    @torch.no_grad()
    def encode_to_indices(self, x) -> torch.Tensor:
        """x: (B, 3, 64, 64) in [0,1] -> (B, 64) long, row-major over the 8x8 grid."""
        z_e = self.encoder(x)
        _, indices, _ = self.quantizer(z_e)
        return indices.flatten(1)

    @torch.no_grad()
    def decode_from_indices(self, indices) -> torch.Tensor:
        """indices: (B, 64) or (B, 8, 8) long -> (B, 3, 64, 64) in [0,1]."""
        if indices.dim() == 2:
            indices = indices.view(-1, self.grid_hw, self.grid_hw)
        z_q = self.quantizer.codebook[indices].permute(0, 3, 1, 2).contiguous()
        return self.decoder(z_q)
