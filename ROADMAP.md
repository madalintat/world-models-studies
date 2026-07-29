# Roadmap

Working checklist. Check things off as you go; add notes inline. Dates are
suggestions assuming evenings and weekends, not deadlines.

## Stage 0: Compression (week 1)

- [ ] Read `stage0_compression/WHY.md`
- [ ] Collect a frame dataset with the random policy (`collect.py`)
- [ ] Write the autoencoder yourself before opening `models.py`
- [ ] Run `train.py --smoke` on the laptop, then a full run on one 5090
- [ ] Do every prediction exercise and both break-it labs
- [ ] Latent traversals: can you find the "steering" direction in latent space?
- [ ] Answer the "you get it when" checklist from memory

## Stage 1: Ha's world model (weeks 2-3)

- [ ] Read WHY.md, then the original paper (worldmodels.github.io)
- [ ] Write the MDN-RNN loss yourself from the paper's description
- [ ] Full run: VAE from stage 0, then MDN-RNN, then CMA-ES controller
- [ ] Watch a dream rollout video next to a real rollout
- [ ] Break-it lab: temperature 0.1 vs 1.0 vs 2.0 in the dream
- [ ] Checklist from memory

## Stage 2: Dreamer (weeks 4-6)

- [ ] WHY.md, then skim DreamerV1 and DreamerV3 papers
- [ ] Write the RSSM posterior/prior step yourself
- [ ] Full run on one 5090: world model, then actor-critic in imagination
- [ ] Compare score against stage 1's controller
- [ ] Break-it labs: kill KL balancing, then kill the stochastic latent
- [ ] Checklist from memory

## Stage 3: Tokens and transformers (weeks 7-9)

- [ ] WHY.md, then the IRIS paper
- [ ] Write the VQ straight-through estimator yourself
- [ ] Full run: VQ-VAE, then the dynamics transformer
- [ ] Measure drift: PSNR of rollout vs ground truth over horizon
- [ ] Break-it labs: codebook collapse, temperature-0 sampling
- [ ] Checklist from memory

## Stage 4: Diffusion forcing (weeks 10-13)

- [ ] WHY.md, flow matching from first principles
- [ ] Write the per-frame noise-level training step yourself
- [ ] Full run on one 5090, then a data-parallel run on the full box
- [ ] Compare drift curve against stage 3 at equal compute
- [ ] Break-it labs: clean context (no noise) at train time, v-pred vs x-pred
- [ ] Checklist from memory

## Stage 5: Frontier (ongoing)

- [ ] Guided read of `../open-dreamer` with the file map in stage5
- [ ] Pick one: WMGym submission, open-dreamer agent loop, or a drift ablation
- [ ] Write up whatever you find, even if negative

## Meta

- [ ] Push your work after each stage
- [ ] Keep NOTEBOOK.md honest
