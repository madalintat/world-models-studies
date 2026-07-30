# Setup

## Laptop (CPU)

```bash
uv sync
uv run pytest            # all stages' smoke tests
uv run python -m stage0_compression.train --smoke
```

The default lockfile pins CPU torch on purpose so smoke runs and tests work
anywhere, fast, with no CUDA fuss.

If `gymnasium[box2d]` fails to build on a fresh machine, install swig first
(`sudo apt install swig` or `uv tool install swig`), then sync again. Box2d
compiles a C++ extension and swig is its one awkward build dependency.

## A CUDA GPU machine

Same repo, swap torch for the CUDA build in a separate venv so the CPU
lockfile stays untouched:

```bash
uv venv .venv-gpu --python 3.12
UV_PROJECT_ENVIRONMENT=.venv-gpu uv pip install torch --index-url https://download.pytorch.org/whl/cu128
UV_PROJECT_ENVIRONMENT=.venv-gpu uv pip install numpy einops imageio tqdm pytest "gymnasium[box2d]"
UV_PROJECT_ENVIRONMENT=.venv-gpu uv run python -c "import torch; print(torch.cuda.get_device_name(0))"
```

No install of the repo itself is needed: every stage runs as
`python -m stageN_xxx.some_module` from the repo root, which puts the repo
on the path.

RTX 5090 is Blackwell (sm_120), which needs a recent torch cu128+ wheel. If
the box already has a system CUDA setup that fights you, a plain
`python -m venv` with the same installs works just as well.

On a shared multi-GPU machine, pin one visible card so you only use what
you need:

```bash
CUDA_VISIBLE_DEVICES=3 UV_PROJECT_ENVIRONMENT=.venv-gpu uv run python -m stage0_compression.train --model vae --device cuda
```

Every training stage takes `--device cuda` for its full run; each stage
README's "Full run" section has the exact commands.

## Modal

The stages do not ship Modal launcher scripts; each stage README carries a
cost estimate instead. To burst a full run to Modal, wrap the stage's train
module in a small app of your own:

```bash
uv tool install modal
modal setup                      # once, links your account
modal run my_modal_run.py        # a function that calls e.g. stage4_diffusion_forcing.train
```

Cost notes live in each stage's docs. Rule of thumb: an A100-40GB is around
$2-3 per hour, so check the stage's estimated hours before launching, and
use checkpoint/resume so an interrupted run does not burn credits twice.

## Data

Every stage collects its own CarRacing data with a deterministic seed and
caches it under `data/` (gitignored). Delete `data/` to force recollection.
