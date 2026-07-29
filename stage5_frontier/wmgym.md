# WM-Gym participation plan

Site: https://wm-gym.labs.reka.ai/

WM-Gym is a public leaderboard for world models. Instead of judging models
by how pretty their rollouts look, it scores them on whether their
predictions are good enough to make decisions with. That framing matches
this course's whole arc, which is why it is the recommended first public
milestone after stage 5.

## The metric, in two sentences

Decision regret asks: if you rank or choose among candidate action
sequences using the rewards your world model predicts, how much real return
do you give up compared to choosing with the true rewards? A model can be
blurry, drifty, and ugly and still score well if its predicted rewards
order actions correctly, and a gorgeous video model scores at chance if
they don't.

## What you have to implement

The interface is a hosted model service that the benchmark queries:

- an init call that receives the initial observation (and task/env id) and
  sets up your model's state;
- a step call that receives an action and returns your model's predicted
  reward for that step (advancing your internal latent state as it goes);
- a /score HTTP endpoint wrapping the above, which the evaluation harness
  hits with observation/action sequences and reads predicted rewards back
  from.

Check the site's current spec before building; the exact payloads have been
evolving. But the shape is stable: state in, actions in, predicted rewards
out, over HTTP.

## Why a reward head is mandatory

Nothing in stages 0, 1, 3, or 4 predicts reward. Those models predict
frames or latents; decision regret is computed entirely from predicted
reward, so a pure video model literally cannot answer the query. Even
open-dreamer's billion-parameter dynamics model has no reward head today (its
dataset config has a reward-biased sampling knob and nothing downstream
consumes reward). You must add a head that maps your latent state to a
scalar per-step reward and train it on real transitions.

This is why stage 2 is the right starting point: Dreamer already needs a
reward predictor to train its critic in imagination, so your stage 2 model
has the head, the recurrent state to hang it on, and the training loop that
fits it. Porting stage 2 to a WM-Gym submission is mostly serving work:
wrap the RSSM posterior update in init/step, return the reward head's
output, put FastAPI or similar in front, containerize.

## Honest read of the leaderboard

As of mid-2026 the Atari board shows a random baseline at 39 percent and
Dreamer-v3 at 35 percent. Sit with that: the random baseline currently
beats the best-known model-based agent family's entry. That is not a joke
about DreamerV3; it means the board is young, entries are few, the metric
is unforgiving of miscalibrated reward scales, and existing submissions
were likely not tuned for this exact protocol. Concretely for you: a
careful, well-calibrated DreamerV3-style submission is a real target, not a
fantasy. Getting a principled model solidly above the random baseline
would be a visible result for a course-trained solo submission.

## Recommended path

1. Start on dm_control tasks, not Atari. Continuous control is what stage
   2's CarRacing setup is closest to, dm_control observations are clean,
   and the robotics branch (robotics.md) needs the same port anyway.
2. Train stage 2's model on the chosen dm_control tasks with the reward
   head fitted on real transitions; validate reward prediction on held-out
   episodes before caring about anything else. Reward calibration (scale
   and bias, per task) is cheap and probably worth more points than model
   size.
3. Build the init/step wrapper and /score endpoint, and test it against a
   local replica of the scoring loop: feed held-out action sequences,
   compute your own decision regret, and only submit once the local number
   beats a random-reward predictor by a clear margin.
4. Submit, then iterate on the one thing the metric rewards: reward
   ranking quality over long horizons, which in practice means fighting
   the same drift you measured all course.

Budget: one dm_control submission end to end is maybe 20-40 GPU-hours of
training plus a weekend of serving glue. It is the cheapest of the stage 5
exits and the only one with a public scoreboard attached.
