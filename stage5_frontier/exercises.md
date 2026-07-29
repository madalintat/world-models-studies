# Stage 5 exercises

Same rules as every stage: write your committed answer down before running
anything. All commands run from the repo root, with open-dreamer checked
out next door at `../open-dreamer` (clone it from
github.com/next-state/open-dreamer).

## Prediction exercises

### P1: which layers are time layers

The rule, from `BlockCausalLayer` in open-dreamer's models.py, is that
layer i does time attention when `(i + time_layer_offset) % time_every == 0`,
and space attention otherwise. The dynamics config uses depth 30,
time_every 4, offset 0; the tokenizer encoder uses depth 12, time_every 4,
offset 3; the tokenizer decoder uses depth 8 with the same offset.

Commit to: how many time layers in each of the three stacks, and which
index is the first time layer of the tokenizer encoder.

```
grep -n "time_every\|time_layer_offset\|depth" ../open-dreamer/configs/dynamics.yaml ../open-dreamer/configs/tokenizer.yaml
uv run python -c "
for name, depth, off in [('dynamics', 30, 0), ('tok-enc', 12, 3), ('tok-dec', 8, 3)]:
    t = [i for i in range(depth) if (i + off) % 4 == 0]
    print(name, 'time layers:', len(t), t)"
```

Expected: dynamics 8 (layers 0, 4, ..., 28), encoder 3 (first at index 1),
decoder 2. If you predicted the dynamics count as 7 or the encoder's first
time layer as 0, you assumed the offset does nothing; it shifts the whole
pattern, and the two models deliberately use different offsets.

### P2: tokens per timestep

The dynamics model packs 512 tokenizer latents by a factor of 2 and adds
one action token, one shortcut token, and 32 registers. The tokenizer
encoder sees frames padded to 368x640 with 16x16 patches, plus its 512
learned latent tokens.

Commit to: S per timestep for the dynamics model, and S per frame for the
tokenizer encoder. Then predict the ratio of time-attention sequence
lengths the KV caches must hold for one frame of dynamics vs one frame of
encoder.

```
grep -n "n_latents\|packing_factor\|n_register" ../open-dreamer/configs/dynamics.yaml ../open-dreamer/configs/tokenizer.yaml
grep -n "^H:\|^W:\|padding_H\|padding_W\|patch_size" ../open-dreamer/configs/dataset/minecraft_vpt.yaml
uv run python -c "
dyn = 1 + 1 + 512 // 2 + 32
enc = 512 + (368 // 16) * (640 // 16)
print('dynamics S:', dyn, ' encoder S:', enc)"
```

Expected: dynamics 290, encoder 1432 (920 patches plus 512 latents). The
point to internalize: the dynamics model runs on 5x fewer tokens per
timestep than the encoder, which is why dynamics can afford 30 layers of
1920 width and a 192-frame window. Compression is compute budget.

### P3: the beta ladder telescopes

`DenoiseSchedule` defines beta[s] = (1 - tau[s+1]) / (1 - tau[s]) and the
sampler updates x to beta*x + (1-beta)*x0_pred.

Commit to: the four beta values for a 4-step schedule, and the value of
the product beta[0]*beta[1]*...*beta[s-1] as a function of tau[s].

```
uv run python -c "
import numpy as np
tau = np.linspace(0, 1, 5)
beta = (1 - tau[1:]) / (1 - tau[:-1] + 1e-9)
print('beta:', np.round(beta, 4))
print('cumprod:', np.round(np.cumprod(beta), 4), ' vs 1-tau:', 1 - tau[1:])"
```

Expected: beta = [0.75, 0.6667, 0.5, 0.0] and the running product equals
1 - tau exactly. Meaning: after s steps, the fraction of the original
noise still in x is exactly 1 - tau[s], regardless of what the model
predicts, and the final step (beta 0) removes the last of it in one go.
The schedule controls contamination; the model only controls where x0
points.

### P4: what enters the KV cache during rollout

During free-running generation, `next_latent` finishes denoising a frame
and then writes it into the dynamics KV cache.

Commit to one of: (a) the clean denoised latent is cached, (b) the latent
is re-noised to roughly 0.9 signal before the caching forward pass, (c)
the last intermediate ladder state is cached.

```
grep -n "tau_ctx" ../open-dreamer/dreamer/generation.py | head -20
```

Expected: (b). You will find tau_ctx_target = 0.9 snapped onto a ladder in
`DenoiseSchedule.init`, and in `next_latent` a `latent_noised_caching`
built as `latent * tau_ctx + (1 - tau_ctx) * noise` right before the final
dynamics call. If you picked (a): that is exactly the train/test mismatch
stage 4 exists to remove; the model was trained with noisy context, so
serving it clean context at inference would be off-distribution.

### P5: batch anatomy after step 100k

`configs/dynamics.yaml` sets B 8, bootstrap_start 100_000, and
bootstrap_fraction 0.25.

Commit to: after step 100k, how many rows of each batch train the
bootstrap loss, which step sizes those rows sample, and whether tau = 1.0
(fully clean frames) can appear in them.

```
grep -n "bootstrap_start\|bootstrap_fraction\|B:" ../open-dreamer/configs/dynamics.yaml
grep -n "include_endpoint" ../open-dreamer/dreamer/training.py | head -5
```

Expected: 2 of 8 rows; they sample coarser-than-minimum step sizes
(`sample_step_excluding_dmin`); and tau = 1.0 is excluded for them
(`include_endpoint=False` in `sample_tau_for_step`), because a bootstrap
row must leave room for two half-steps above its tau. The remaining 6 rows
are plain diffusion forcing at the finest step, clean frames included.

## Break-it labs

### B1: rot the guide, watch the checker catch it

Sabotage: rename a real symbol in a copy of the guide, then run the
checker against the copy.

```
cp stage5_frontier/guide_open_dreamer.md /tmp/guide_sabotaged.md
sed -i 's/KVCache/KVCacheRing/g' /tmp/guide_sabotaged.md
uv run python -m stage5_frontier.check_refs --guide /tmp/guide_sabotaged.md; echo "exit: $?"
```

What to observe: the checker reports
`missing symbol: dreamer/models.py::KVCacheRing` and exits 1, while all
untouched references still pass. Then try a subtler sabotage: change a
path instead (`sed -i 's|dreamer/sampler.py|dreamer/sampling.py|g'`) and
rerun; you get `missing file` instead of `missing symbol`.

What it teaches: documentation drifts in two distinct ways (files move,
symbols get renamed) and a checker must catch both. Also the checker's
limits: it verifies existence, not truth. A sentence that cites the right
symbol and describes it wrongly sails through. Existence checks are the
floor of grounding, not the ceiling; the ceiling is you, reading.

### B2: run the tau ladder one step short

Replicate the sampler's mixing rule in numpy with a perfect model (the
x0 prediction is always exactly the target), then sabotage it by skipping
the final step.

```
uv run python - <<'EOF'
import numpy as np
rng = np.random.default_rng(0)
target = 2.0
tau = np.linspace(0, 1, 5)
beta = (1 - tau[1:]) / (1 - tau[:-1] + 1e-9)

x = rng.normal()
x_start = x
for b in beta:                 # full ladder
    x = b * x + (1 - b) * target
print("full ladder:   ", x, "(target", target, ")")

x = x_start
for b in beta[:-1]:            # sabotage: skip the last step
    x = b * x + (1 - b) * target
print("one step short:", x, "= 0.25*start + 0.75*target:", 0.25 * x_start + 0.75 * target)
EOF
```

What to observe: the full ladder lands on the target exactly, even from
random noise, because the last beta is 0. One step short leaves exactly 25
percent of the starting noise in the sample, matching the telescoping rule
from P3 (tau reached is 0.75, so 1 - 0.75 of the noise remains).

What it teaches: with an x-prediction sampler, the model never "cleans"
anything; the schedule does, deterministically. Any bug that misaligns
schedule and step count (wrong num_steps, off-by-one in the scan, betas
from a different ladder) leaves a precise, computable fraction of noise in
every generated frame, which then gets fed back as context and compounds.
When a diffusion world model produces uniformly hazy rollouts, check the
schedule before the weights.

### B3: read the ring buffer without rolling

Replicate `KVCache`'s ring buffer in numpy, then sabotage the read path by
skipping the roll that `get_ordered_kv` performs.

```
uv run python - <<'EOF'
import numpy as np
window = 4
buf = np.zeros(window)
idx = 0
for v in [1, 2, 3, 4, 5, 6]:   # write 6 frames into a window of 4
    buf[idx % window] = v
    idx += 1
print("raw buffer:    ", buf)
print("rolled (correct):", np.roll(buf, -(idx % window)))
EOF
```

What to observe: the raw buffer reads [5, 6, 3, 4]; the rolled view reads
[3, 4, 5, 6]. The sabotaged version presents frame 5 as the *oldest* key
and frame 4 as the newest.

What it teaches: a ring buffer stores a window, not an order; order is
reconstructed at read time. Time attention with RoPE assigns positions by
sequence index, so the unrolled read tells the model that the two most
recent frames are the two most distant ones. Nothing crashes, shapes are
all fine, and the rollout just smears whenever the write index wraps,
which is the worst kind of bug: intermittent, shape-silent, and appearing
only after exactly `window` frames. Now reread `get_ordered_kv` in
open-dreamer's models.py and notice it also masks the zero-filled slots
during warmup; that is the same class of bug, pre-killed.
