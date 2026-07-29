import numpy as np
import pytest
import torch

from stage3_token_transformer.rollout3 import compute_drift, rollout
from stage3_token_transformer.s3_transformer import (
    IGNORE_INDEX,
    GPTConfig,
    TokenGPT,
    make_targets,
    sequence_loss,
)
from stage3_token_transformer.vqvae import VQVAE, VectorQuantizerEMA

torch.manual_seed(0)
# Small tensors thrash on many-core CPUs when torch grabs every thread.
torch.set_num_threads(4)


def tiny_gpt(max_frames=4):
    cfg = GPTConfig(d_model=32, n_heads=2, n_layers=2, max_frames=max_frames, dropout=0.0)
    return TokenGPT(cfg).eval()


def test_quantizer_shapes_and_gradient_flow():
    model = VQVAE(base=16)
    x = torch.rand(2, 3, 64, 64)
    z_e = model.encoder(x)
    assert z_e.shape == (2, 64, 8, 8)
    z_q, indices, info = model.quantizer(z_e)
    assert z_q.shape == (2, 64, 8, 8)
    assert indices.shape == (2, 8, 8)
    assert indices.dtype == torch.long
    loss, _ = model.loss(x)
    loss.backward()
    first_conv = model.encoder.net[0].weight
    assert first_conv.grad is not None
    assert first_conv.grad.abs().sum() > 0, "straight-through gradient did not reach the encoder"


def test_codebook_usage_tracked_and_ema_moves():
    q = VectorQuantizerEMA(num_codes=32, code_dim=8, decay=0.9)
    q.train()
    before = q.codebook.clone()
    z = torch.randn(4, 8, 4, 4)
    q(z)
    assert q.usage.sum().item() == 4 * 4 * 4
    hist = q.usage_histogram()
    assert hist.shape == (32,)
    assert torch.isclose(hist.sum(), torch.tensor(1.0))
    assert not torch.allclose(before, q.codebook), "EMA update did not move the codebook"

    frozen = VectorQuantizerEMA(num_codes=32, code_dim=8, enable_ema=False, enable_dead_reinit=False)
    frozen.train()
    before = frozen.codebook.clone()
    frozen(z)
    assert torch.allclose(before, frozen.codebook)


def test_dead_code_reinit_replaces_dead_rows():
    q = VectorQuantizerEMA(num_codes=16, code_dim=4, dead_threshold=0.2)
    q.ema_cluster_size[:8] = 0.01
    dead_rows_before = q.codebook[:8].clone()
    flat = torch.randn(100, 4)
    q._reinit_dead(flat)
    assert not torch.allclose(dead_rows_before, q.codebook[:8])
    assert (q.ema_cluster_size[:8] == 1.0).all()
    # Reinitialized rows must be actual encoder outputs from the batch.
    for row in q.codebook[:8]:
        assert (flat == row).all(dim=1).any()


def test_make_targets_ignores_action_slots():
    tokens = torch.arange(2 * 3 * 4).view(2, 3, 4) % 256
    tgt = make_targets(tokens)
    assert tgt.shape == (2, 3 * 5)
    # Position 4 is the last z token of frame 0; its successor is action a_1.
    assert (tgt[:, 4] == IGNORE_INDEX).all()
    assert (tgt[:, -1] == IGNORE_INDEX).all()
    # Position 0 (a_0) predicts the first z token of frame 0.
    assert (tgt[:, 0] == tokens[:, 0, 0]).all()


def test_causal_mask_future_does_not_leak():
    model = tiny_gpt()
    actions = torch.randn(1, 3, 3)
    tokens = torch.randint(0, 256, (1, 3, 64))
    logits_a = model(actions, tokens)
    tokens_b = tokens.clone()
    tokens_b[:, -1] = torch.randint(0, 256, (1, 64))
    logits_b = model(actions, tokens_b)
    cut = 2 * 65 + 1  # everything up to and including a_2 is unchanged input
    assert torch.allclose(logits_a[:, :cut], logits_b[:, :cut], atol=1e-5)
    assert not torch.allclose(logits_a[:, cut:], logits_b[:, cut:], atol=1e-5)


def test_kv_cache_matches_full_forward():
    model = tiny_gpt()
    actions = torch.randn(1, 2, 3)
    tokens = torch.randint(0, 256, (1, 2, 64))
    emb = model.embed_sequence(actions, tokens)
    full_logits, _ = model.step(emb)

    kv = None
    step_logits = []
    for i in range(emb.shape[1]):
        lg, kv = model.step(emb[:, i : i + 1], kv)
        step_logits.append(lg)
    step_logits = torch.cat(step_logits, dim=1)
    assert torch.allclose(full_logits, step_logits, atol=1e-4)

    # Also check a chunked prefill (more than one new token with a nonempty cache).
    lg_head, kv = model.step(emb[:, :3])
    lg_tail, _ = model.step(emb[:, 3:10], kv)
    assert torch.allclose(full_logits[:, 3:10], lg_tail, atol=1e-4)


def test_dyn_loss_decreases_on_fixed_batch():
    model = tiny_gpt().train()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    actions = torch.randn(2, 3, 3)
    tokens = torch.randint(0, 256, (2, 3, 64))
    losses = []
    for _ in range(30):
        loss = sequence_loss(model(actions, tokens), tokens)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0] * 0.9


def test_vq_loss_decreases_on_fixed_batch():
    model = VQVAE(base=16)
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    x = torch.rand(8, 3, 64, 64)
    losses = []
    for _ in range(20):
        loss, _ = model.loss(x)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0]


def test_drift_returns_finite_psnr():
    rng = np.random.default_rng(0)
    a = rng.integers(0, 256, size=(5, 64, 64, 3), dtype=np.uint8)
    b = rng.integers(0, 256, size=(5, 64, 64, 3), dtype=np.uint8)
    psnrs = compute_drift(a, b)
    assert len(psnrs) == 5
    assert all(np.isfinite(p) for p in psnrs)
    same = compute_drift(a, a.copy())
    assert all(np.isfinite(p) for p in same)


def test_rollout_shapes_with_untrained_models():
    vq = VQVAE(base=16).eval()
    gpt = tiny_gpt(max_frames=5)
    ctx = np.random.default_rng(1).integers(0, 256, size=(2, 64, 64, 3), dtype=np.uint8)
    acts = np.zeros((5, 3), dtype=np.float32)
    gen, toks = rollout(vq, gpt, ctx, acts, n_future=3, temperature=1.0)
    assert gen.shape == (3, 64, 64, 3)
    assert gen.dtype == np.uint8
    assert toks.shape == (3, 64)
    assert toks.min() >= 0 and toks.max() < 256
    gen0, _ = rollout(vq, gpt, ctx, acts, n_future=3, temperature=0.0)
    gen0b, _ = rollout(vq, gpt, ctx, acts, n_future=3, temperature=0.0)
    assert np.array_equal(gen0, gen0b), "temperature 0 rollouts should be deterministic"
