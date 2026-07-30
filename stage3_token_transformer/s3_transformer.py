"""Small GPT over interleaved action and frame tokens.

Sequence layout: [a_0, z_0^1..z_0^64, a_1, z_1^1..z_1^64, ...]. Each action is
one token produced by a linear projection of the continuous (steer, gas, brake)
vector. Each frame contributes 64 code indices from the VQ-VAE, row-major over
the 8x8 grid. The model predicts the next element at every position; the loss
only counts positions whose next element is a frame token, because actions are
given by the controller, never predicted.

Learned positional embeddings. KV cache supported through step().
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

IGNORE_INDEX = -100


@dataclass
class GPTConfig:
    vocab_size: int = 256
    tokens_per_frame: int = 64
    action_dim: int = 3
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 4
    max_frames: int = 16

    @property
    def frame_block(self) -> int:
        return self.tokens_per_frame + 1

    @property
    def block_size(self) -> int:
        return self.max_frames * self.frame_block


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        assert cfg.d_model % cfg.n_heads == 0
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.d_model // cfg.n_heads
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model)

    def forward(self, x, past_kv=None):
        B, L, D = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)

        def heads(t):
            return t.view(B, L, self.n_heads, self.head_dim).transpose(1, 2)

        q, k, v = heads(q), heads(k), heads(v)
        past_len = 0
        if past_kv is not None:
            pk, pv = past_kv
            past_len = pk.shape[2]
            k = torch.cat([pk, k], dim=2)
            v = torch.cat([pv, v], dim=2)
        if past_len == 0:
            y = F.scaled_dot_product_attention(
                q, k, v, is_causal=True
            )
        else:
            # New queries see the whole cache plus a causal pattern among themselves.
            total = past_len + L
            i = torch.arange(L, device=x.device).unsqueeze(1)
            j = torch.arange(total, device=x.device).unsqueeze(0)
            mask = j <= (past_len + i)
            y = F.scaled_dot_product_attention(
                q, k, v, attn_mask=mask
            )
        y = y.transpose(1, 2).contiguous().view(B, L, D)
        return self.proj(y), (k, v)


class Block(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.d_model, 4 * cfg.d_model),
            nn.GELU(),
            nn.Linear(4 * cfg.d_model, cfg.d_model),
        )

    def forward(self, x, past_kv=None):
        a, kv = self.attn(self.ln1(x), past_kv)
        x = x + a
        x = x + self.mlp(self.ln2(x))
        return x, kv


class TokenGPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.act_emb = nn.Linear(cfg.action_dim, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.d_model)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layers))
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

    def embed_sequence(self, actions, tokens):
        """actions (B,T,3) float, tokens (B,T,K) long -> (B, T*(K+1), D)."""
        a = self.act_emb(actions).unsqueeze(2)
        z = self.tok_emb(tokens)
        return torch.cat([a, z], dim=2).flatten(1, 2)

    def step(self, x_emb, past_kv=None):
        """Run embedded inputs through the stack, appending to the KV cache.

        x_emb: (B, L, D). past_kv: list of per-layer (k, v) or None.
        Returns (logits (B, L, vocab), new_kv).
        """
        B, L, _ = x_emb.shape
        past_len = 0 if past_kv is None else past_kv[0][0].shape[2]
        assert past_len + L <= self.cfg.block_size, "sequence exceeds block_size"
        pos = torch.arange(past_len, past_len + L, device=x_emb.device)
        x = x_emb + self.pos_emb(pos)
        new_kv = []
        for i, blk in enumerate(self.blocks):
            x, kv = blk(x, None if past_kv is None else past_kv[i])
            new_kv.append(kv)
        return self.head(self.ln_f(x)), new_kv

    def forward(self, actions, tokens):
        logits, _ = self.step(self.embed_sequence(actions, tokens))
        return logits


def make_targets(tokens: torch.Tensor) -> torch.Tensor:
    """tokens (B,T,K) -> next-element targets (B, T*(K+1)) with IGNORE_INDEX
    at positions whose successor is an action token (or does not exist)."""
    B, T, K = tokens.shape
    seq = torch.full((B, T, K + 1), IGNORE_INDEX, dtype=torch.long, device=tokens.device)
    seq[:, :, 1:] = tokens
    seq = seq.flatten(1)
    targets = torch.full_like(seq, IGNORE_INDEX)
    targets[:, :-1] = seq[:, 1:]
    return targets


def sequence_loss(logits: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
    targets = make_targets(tokens)
    return F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), ignore_index=IGNORE_INDEX
    )
