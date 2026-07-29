# Three experiments that fit on a single 8-GPU workstation

These are real experiment specs, not homework. Each one has a question the
field does not have a crisp public answer to at this scale, a concrete
procedure using code you already built, a GPU-hour budget, and an honest
statement of what each outcome would mean. All numbers assume eight RTX
5090 class cards; a 5090-hour here is roughly an A100-hour for these model
sizes. Write up whatever you find, including nulls. A clean negative result
with error bars is a contribution.

Ground rule for all three: fix the CarRacing-v3 dataset once (same seeds,
same episodes, frozen on disk) and reuse it everywhere. Any comparison where
the data differs is dead on arrival.

## Experiment 1: drift benchmark, stage 3 vs stage 4 at matched compute

The claim behind stage 4's existence is that training with noisy context
makes autoregressive rollouts degrade more slowly than next-token prediction
does. You saw it qualitatively. The experiment is to measure it under a
fair budget, because the honest version of the question is not "is diffusion
forcing better" but "is it better per FLOP".

Procedure:

1. Freeze the dataset: 500 CarRacing episodes, fixed seeds, 64x64, the
   same collection script both stages already use.
2. Compute the FLOPs per training step of the stage 3 dynamics transformer
   and the stage 4 model analytically (the estimate_flops methods in
   open-dreamer show the accounting style; replicate it for your models).
   Pick step counts so total training FLOPs match within 5 percent. Keep
   the tokenizer identical between arms if at all possible (stage 4 can
   consume the stage 3 VQ latents' pre-quantization embeddings); if not,
   report tokenizer reconstruction PSNR for both so the confound is visible.
3. Train 3 seeds per arm. Suggested full config per run: batch 64 sequences
   of 32 frames, 60k steps, one card per run, so all 6 runs fit the box in
   parallel.
4. Evaluate identically: 100 held-out contexts of 16 frames, roll out 200
   frames with ground-truth actions, report PSNR and LPIPS versus horizon,
   mean and seed spread. Also report frames-to-threshold: the horizon at
   which PSNR first drops below 18 dB, which reads better than curve
   eyeballing.
5. One ablation that makes the writeup: rerun the stage 4 arm with clean
   (noise-free) context at train time, same budget. That isolates whether
   the win comes from noisy context specifically or from the denoising
   objective generally.

Budget: 6 main runs plus 3 ablation runs at about 12 GPU-hours each, plus
about 10 GPU-hours of evaluation: roughly 120-130 GPU-hours, about 17 hours
of wall clock on 8 cards.

Deliverable: one plot (PSNR vs horizon, three curves: stage 3, stage 4,
stage 4 clean-context, with seed bands) and a 1-2 page writeup.

What results would mean: if stage 4 decays clearly slower at matched
compute, you have a small, clean, citable confirmation of the diffusion
forcing story on a public environment. If the curves overlap, that is more
interesting: it suggests drift at this scale is dominated by the tokenizer
or by CarRacing's simplicity, and the noisy-context story needs bigger
worlds to show up. If the clean-context ablation matches the full stage 4
arm, the field's favorite explanation for why these models drift less is
doing less work than advertised. Every branch is worth writing up.

## Experiment 2: memory probe, does the world model remember off-screen track

CarRacing's camera follows the car. A corner that scrolled off the top of
the frame still exists, and when the car comes back around, a good world
model should redraw the same corner, not a freshly hallucinated one. Sliding
window attention (open-dreamer caps time attention at 192 frames; your stage
4 model has some window too) puts a hard ceiling on this. The question:
within the window, does the model actually use its memory, or does it just
redraw plausible track?

Procedure:

1. Build a probe set. Collect episodes with a scripted controller that
   drives forward, and locate re-entry events: frame pairs (t_leave,
   t_return) where a distinctive track feature (a sharp corner) exits the
   camera view and later re-enters. CarRacing gives you the true track
   polygon in the env internals, so you can find these events exactly
   rather than by pixel heuristics. Keep events with gap lengths spread
   over 10 to 150 frames.
2. Metric. At t_return, segment track versus grass by color threshold
   (gray versus green, trivially separable in CarRacing) in both the
   predicted frame and the ground-truth frame, and compute IoU of the
   track masks. Call it re-entry IoU. Baseline: permutation IoU, the same
   score computed against the track mask from a *different* re-entry event
   with matched car pose. A model with zero layout memory but perfect
   track-drawing skill scores at the permutation baseline; memory is the
   gap above it.
3. Roll out the stage 4 model from context ending shortly before t_leave,
   with ground-truth actions, through t_return. Score re-entry IoU as a
   function of the off-screen gap length.
4. Train the modification: three stage 4 retrains differing only in
   temporal context window: 8, 32, and 128 frames (batch 32, 40k steps
   each). Prediction to commit to before running: re-entry IoU should sit
   at the permutation baseline whenever the gap exceeds the window, and
   rise above it when the window covers the gap. If it does not rise even
   then, the model has the capacity for memory but the training objective
   never forced it to use it, which is the more damning finding.

Budget: 3 retrains at about 15 GPU-hours plus probe-set construction and
evaluation at about 10 GPU-hours: roughly 55-60 GPU-hours, under a day on
the box.

Deliverable: re-entry IoU vs gap length, one curve per context window, with
the permutation baseline as a horizontal band, plus a gallery of the worst
hallucinated corners.

What results would mean: a clear window-dependent memory signal says these
models do store and reuse spatial layout, and quantifies the horizon. A
flat curve at baseline says pixel-space video models fake persistence, which
is exactly the criticism the Genie 3 line of work is trying to bury, and a
concrete metric for it on a free environment is genuinely useful to other
people. Either way you have built a memory benchmark, and it ports to any
world model that runs on CarRacing.

## Experiment 3: dream-cheating study, actor quality vs dream quality

Stage 1's controller cheated the blurry RNN dream; stage 2's actor was
supposed to cheat less because the dream was better. Nobody in the course
measured the relationship. The question: how does the gap between imagined
return and real return shrink as world-model quality improves, and does it
ever reach zero?

Procedure:

1. Produce a ladder of world models of controlled quality using stage 2's
   code: train the RSSM world model and save checkpoints at 5, 10, 25, 50,
   and 100 percent of full training, and add one extra arm trained on 10x
   less data at full steps. That is 6 quality levels, 2 seeds each, 12
   world models. Record each one's observation reconstruction loss and
   50-step rollout PSNR as the "dream quality" axis.
2. For each frozen world model, train the stage 2 actor-critic purely in
   imagination, identical hyperparameters, identical step budget. Log the
   imagined return the actor believes it achieves.
3. Evaluate every actor in the real CarRacing env, 50 episodes, fixed seed
   set. The headline quantity is the cheat gap: imagined return minus real
   return, plotted against dream quality.
4. Secondary analysis, cheap and revealing: for the worst and best world
   models, take the states the actor visits in dreams and decode them.
   Cheating has a signature you already know from stage 1: the actor
   steers into regions where the dream goes soft and reward stays high.
   Two videos side by side make the writeup.

Budget: 12 world-model trainings at about 4 GPU-hours, 12 actor trainings
at about 2 GPU-hours, evaluation is CPU-bound and near free: roughly 75-80
GPU-hours, about 10 hours on the box.

Deliverable: cheat gap vs dream quality scatter with both seeds, the two
dream videos, and a short writeup.

What results would mean: a monotone gap that approaches zero supports the
core bet of the whole Dreamer line: buy a better dream, get a better
policy for free. A gap that plateaus above zero at high quality is the
sharper result: it says actors find adversarial directions faster than
world models close them, which is the strongest argument for the
model-quality-first strategy of Dreamer 4 (train enormous world models,
keep the policy learning modest) and also for regularizers that penalize
the actor for leaving the model's competence region. A non-monotone gap
would mean dream quality metrics like PSNR do not track exploitability at
all, which would itself justify a follow-up.

## Total budget

All three experiments: roughly 270 GPU-hours, about 2.5 days of the full
box, or one long week running them alongside other work. Modal cost note:
on rented A100-40GB at $1.10-1.90/h this is about $300-500 end to end; run
smokes locally first so none of that budget buys typos.
