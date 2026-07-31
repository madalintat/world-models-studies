"""Temporal transformer for latent-space diffusion forcing.

Per frame the token sequence is [action token, tau token, 64 latent tokens].
Attention is factorized: space layers run full attention over the tokens of
one frame, time layers run causal attention across frames at the same token
position, and the two alternate through the depth. open-dreamer's
BlockCausalTransformer (dreamer/models.py) factorizes the same way, with one
time layer every four layers instead of strict alternation.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


def sinusoidal_tau_embedding(tau: torch.Tensor, dim: int) -> torch.Tensor:
    """tau in [0, 1], shape (B, T) -> (B, T, dim). Frequencies span 1 to 1000
    so nearby noise levels stay distinguishable."""
    half = dim // 2
    freqs = torch.exp(
        torch.linspace(math.log(1.0), math.log(1000.0), half, device=tau.device)
    )
    args = tau.unsqueeze(-1) * freqs
    return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class AxialBlock(nn.Module):
    """Pre-LN transformer block attending along one axis of (B, T, S, D).

    With injection="film" the block also carries a per-frame FiLM
    modulation applied between attention and the MLP; see
    DynamicsTransformer for what the injection choice means.
    """

    def __init__(self, d_model: int, n_heads: int, axis: str, injection: str = "token"):
        super().__init__()
        assert axis in ("space", "time")
        self.axis = axis
        self.n_heads = n_heads
        self.injection = injection
        self.norm1 = nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )
        if injection == "film":
            self.film = nn.Linear(d_model, 2 * d_model)
            # Zero init so the block starts as the identity modulation and
            # FiLM cannot destabilize early training.
            nn.init.zeros_(self.film.weight)
            nn.init.zeros_(self.film.bias)

    def forward(self, x: torch.Tensor, act: torch.Tensor | None = None) -> torch.Tensor:
        b = x.shape[0]
        h = self.norm1(x)
        if self.axis == "space":
            h = rearrange(h, "b t s d -> (b t) s d")
        else:
            h = rearrange(h, "b t s d -> (b s) t d")
        q, k, v = rearrange(
            self.qkv(h), "n l (three heads e) -> three n heads l e",
            three=3, heads=self.n_heads,
        )
        a = F.scaled_dot_product_attention(q, k, v, is_causal=(self.axis == "time"))
        a = self.proj(rearrange(a, "n heads l e -> n l (heads e)"))
        if self.axis == "space":
            a = rearrange(a, "(b t) s d -> b t s d", b=b)
        else:
            a = rearrange(a, "(b s) t d -> b t s d", b=b)
        x = x + a
        h2 = self.norm2(x)
        if self.injection == "film" and act is not None:
            # act is (B, T, 1, D): one modulation per frame, shared by every
            # token of that frame.
            gamma, beta = self.film(act).chunk(2, dim=-1)
            h2 = h2 * (1.0 + gamma) + beta
        return x + self.mlp(h2)


class DynamicsTransformer(nn.Module):
    def __init__(
        self,
        latent_dim: int = 8,
        n_latent_tokens: int = 64,
        d_model: int = 128,
        n_heads: int = 4,
        depth: int = 6,
        action_dim: int = 3,
        max_t: int = 32,
        injection: str = "token",
    ):
        """injection decides how the action reaches the network:

        "token"    the action embedding is an extra token in each frame's
                   space sequence, so the latent tokens reach it through
                   attention. This is the course default and the most
                   expensive per frame: it costs a sequence slot forever.
        "additive" the action embedding is added to every latent token of
                   its frame, once, before the blocks. Zero extra
                   parameters and no sequence cost.
        "film"     every block scales and shifts its MLP input by a
                   per-frame affine map predicted from the action, so the
                   action modulates computation at every depth instead of
                   being one more thing to attend to.

        nano-world-model ablates these plus adaLN and cross-attention, and
        finds additive best for low-dimensional actions (all five within
        0.32 PSNR on 2D PushT actions), FiLM slightly ahead on 7D robot
        actions, and cross-attention consistently worst. CarRacing's action
        is 3 numbers, which puts this stage squarely in the regime where
        the cheap option should win. Exercise P5 asks you to commit to a
        prediction before checking.
        """
        super().__init__()
        assert injection in ("token", "additive", "film")
        self.latent_dim = latent_dim
        self.n_latent_tokens = n_latent_tokens
        self.max_t = max_t
        self.injection = injection
        self.latent_in = nn.Linear(latent_dim, d_model)
        self.action_in = nn.Linear(action_dim, d_model)
        self.tau_mlp = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, d_model)
        )
        self.d_model = d_model
        # The action only occupies a sequence slot in "token" mode; the tau
        # token is always there, because the noise level has to reach the
        # network somehow and that choice is not what we are ablating.
        self.n_prefix = 2 if injection == "token" else 1
        n_tokens = n_latent_tokens + self.n_prefix
        self.space_pos = nn.Parameter(torch.randn(1, 1, n_tokens, d_model) * 0.02)
        self.time_pos = nn.Parameter(torch.randn(1, max_t, 1, d_model) * 0.02)
        self.blocks = nn.ModuleList(
            [
                AxialBlock(
                    d_model, n_heads, "space" if i % 2 == 0 else "time", injection
                )
                for i in range(depth)
            ]
        )
        self.out_norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, latent_dim)

    def forward(
        self, z: torch.Tensor, actions: torch.Tensor, tau: torch.Tensor
    ) -> torch.Tensor:
        """z: (B, T, S, latent_dim) noisy latents. actions: (B, T, action_dim).
        tau: (B, T). Returns the x-prediction, (B, T, S, latent_dim)."""
        b, t, s, _ = z.shape
        assert t <= self.max_t, f"sequence length {t} exceeds max_t {self.max_t}"
        assert s == self.n_latent_tokens
        lat = self.latent_in(z)
        act = self.action_in(actions).unsqueeze(2)  # (B, T, 1, D)
        tau_tok = self.tau_mlp(sinusoidal_tau_embedding(tau, self.d_model)).unsqueeze(2)
        if self.injection == "token":
            x = torch.cat([act, tau_tok, lat], dim=2)
        else:
            if self.injection == "additive":
                lat = lat + act
            x = torch.cat([tau_tok, lat], dim=2)
        x = x + self.space_pos + self.time_pos[:, :t]
        for blk in self.blocks:
            x = blk(x, act)
        return self.head(self.out_norm(x[:, :, self.n_prefix :]))
