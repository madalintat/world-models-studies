# Why stage 5 exists

## The core idea

There is no new model in this stage, because the thing being trained is
you. Stages 0 through 4 each handed you one idea and one failure. Stage 5's
claim is that after those four failures, the actual frontier (a 1.6B
parameter Dreamer 4 pipeline, a public decision-making benchmark, the
robotics literature) is not above your head; it is made of parts you have
personally built and broken. The work of this stage is to verify that claim
against reality: read a production codebase and find every course concept
in it, spec experiments that would produce knowledge rather than vibes, and
identify where a contribution you could make is genuinely wanted.

The skill being installed is a specific one: grounding. Every claim in this
stage's documents is checkable, most of them mechanically. That is the
habit that separates research from tourism.

## Design choices and their whys

**A reading tour instead of a summary.** guide_open_dreamer.md maps files
and symbols, not concepts to concepts, because summaries let you nod along
without opening anything. The tour only works with the repo in a second
window, which is the point. Ordering follows dependency (TokenLayout before
models, models before training, training before sampling, configs last)
rather than the directory listing, because that is the order in which the
code explains itself.

**Symbol references, no line numbers.** Line numbers are precise and wrong
within a month. File paths plus class/function names are almost as precise
and survive refactors. `check_refs.py` mechanically verifies every
reference against the checkout, so the guide fails loudly instead of
rotting silently. This is also a quiet demonstration of the stage's thesis:
documentation about code should be tested like code.

**A checker script as the only code.** Stage 5 ships one script and it
verifies documents. That inversion is deliberate: at the frontier, most of
what you produce is claims, and claims need harnesses too. The train.py
entry point exists only so the course-wide smoke command works in every
stage; it runs the checker and costs a second.

**Experiments sized for the box you actually have.** research.md specs
three experiments at 55 to 130 GPU-hours each because that is what 8x5090
delivers in a day or two. Every spec commits to a metric before proposing a
run, states what a null result would mean, and reuses course code. Refusing
to spec beyond your hardware is not modesty; unrunnable experiment designs
are how people stay spectators.

**Three exits, not one.** Research (research.md), contribution
(open-dreamer's missing agent loop), and application (wmgym.md,
robotics.md) are all mapped because people leave a course like this in
different directions, and each document ends in a first concrete step
rather than a vision.

**WM-Gym before robotics hardware.** The benchmark plan front-runs the
robotics ladder because it is the cheapest exit with an external
scoreboard, and because its metric (decision regret) is the course's thesis
stated as an evaluation: a world model is worth exactly as much as the
decisions it supports.

## Common failure modes

- Reading open-dreamer without the mapping discipline. You skim 8k lines,
  feel a warm sense of recognition, and retain nothing. Antidote: for every
  stop in the guide, write the stage number and the one-sentence delta in
  your NOTEBOOK.md before moving on, and do the exercises, which force
  predictions with checkable answers.
- Treating JAX as the content. You can burn a week learning nnx idioms and
  pytree registration. The ideas transfer; the framework is scenery. Learn
  exactly as much JAX as reading requires and no more.
- Speccing experiments you cannot run. The failure looks like ambition and
  functions as avoidance: a 10,000 GPU-hour plan produces zero plots.
  Every spec here fits the box on purpose; scale down further before you
  scale up.
- Leaderboard-chasing without calibration. Submitting a frame predictor to
  a decision-regret benchmark, or an uncalibrated reward head, scores at
  or below the random baseline and teaches nothing. Validate reward
  ranking locally first; wmgym.md walks through it.
- Trusting your own guide. The upstream repo moves. If check_refs starts
  failing, the correct response is to fix the guide against the new code,
  which is itself the stage's exercise repeated.

## Backward and forward

Backward: every stop in the reading tour cites the stage that taught it.
Stage 0 is the tokenizer's compression bet, stage 1 is the dream and its
cheatability, stage 2 is the missing agent loop and the reward head that
WM-Gym demands, stage 3 is tokens, causal attention, KV caching, and the
drift metric, stage 4 is the entire dynamics-training and sampling stack.
If any stop in the guide feels like new material rather than recognition,
that is a signal to go back one stage and redo its break-it labs.

Forward: there is no stage 6; the three exits are the forward edge. The
concrete next actions, pick one: run experiment 1 from research.md and
publish the plot; build the dm_control port that both wmgym.md and
robotics.md need; or prototype the open-dreamer agent loop (stage 2's
actor-critic against stops 5 through 7 of the guide) and open the
conversation upstream.

## Write it yourself first

The file to produce before reading mine: **guide_open_dreamer.md**. Before
opening my tour, spend one hour with `../open-dreamer` and write your own
map: for each of `dreamer/utils.py`, `dreamer/models.py`,
`dreamer/training.py`, `dreamer/generation.py`, and the two main configs,
one line naming the course stage it corresponds to and what looks new.
Then diff against mine. The disagreements are the curriculum; where we
agree you are done, and where we differ, one of us has a checkable claim
and you should check it. (If you want to write code instead, reimplement
`check_refs.py` from its docstring; it is a one-hour exercise in making
documents falsifiable.)

## You get it when

1. For each of the guide's eight stops, you can name the course stage it
   comes from and the one thing that changed at scale, without opening
   the guide.
2. You can explain why open-dreamer's tokenizer has no VQ codebook while
   stage 3's did, in terms of what the downstream dynamics model needs.
3. You can state what tau_ctx is, what value it takes, and why finalized
   frames are re-noised before entering the KV cache during rollout.
4. You can explain what the bootstrap rows in shortcut_forcing_step train,
   why they need two half-steps from an EMA model, and what capability
   the model gains from them at inference time.
5. You can say precisely which stage-2 components are missing from
   open-dreamer, and point to three places in the code where their
   arrival is already plumbed for.
6. You can define decision regret in two sentences and explain why a
   perfect video model with no reward head scores at chance.
7. For each of the three research.md experiments, you can state the
   metric and what a null result would mean, which is different from
   what a positive result would mean.
8. Someone hands you a new world-model repo, and your first three moves
   are: find the tokenizer's bottleneck, find the noise (or token)
   schedule in the training step, find how context enters the sampler.
   You know why those three.
