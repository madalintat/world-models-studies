# The MuJoCo branch

CarRacing was the right teaching environment: one env, pixels only, simple
actions, free. Robotics is where world models are heading commercially, and
the transfer is more mechanical than it looks. This is the ladder to climb,
what actually changes, and what of your code survives.

## The ladder

Climb in this order; each rung adds exactly one kind of difficulty.

1. dm_control (MuJoCo classics: cartpole, walker, cheetah, quadruped).
   Same single-env, dense-reward regime as CarRacing, but with
   proprioception and torque actions. This is where DreamerV1 and PlaNet
   proved out, so you are walking a well-marked trail and can sanity-check
   your scores against published curves. Goal for this rung: stage 2 ported,
   walker-walk solved from pixels plus proprio.
2. Meta-World (Sawyer arm, 50 manipulation tasks). Adds task variety and
   goal-conditioning: one arm, many objectives (push, pick-place, open
   drawer). This is the first place a *shared* world model across tasks
   pays off, which is the actual thesis of world models. Goal: multi-task
   world model, per-task actors, report the standard success-rate metric on
   MT10 before attempting MT50.
3. MJX / ManiSkill. Scale rung. MJX is MuJoCo on accelerator: thousands of
   parallel simulated environments on one GPU, which changes the data
   economics completely (collection stops being the bottleneck; your
   dataloader becomes one). ManiSkill brings GPU-parallel manipulation with
   photoreal-ish rendering and big task suites. This is where the 8x5090
   box starts behaving like the open-dreamer regime: data is effectively
   infinite and compute allocation is the design problem.

## What changes vs CarRacing

Proprioception. Robots know their joint angles and velocities exactly; only
the world outside the body needs vision. Concretely: the observation is now
a dict (pixels plus a low-dimensional vector), and the correct move is an
encoder per modality with fused features, not rendering everything into
pixels. Proprio is low-noise and high-value, so models that get it as a
separate channel train visibly faster. Note open-dreamer already reserves a
PROPRIO modality in its token layout; the frontier codebase you just read
is structured for exactly this.

Rewards. CarRacing hands you a dense scalar for track progress. dm_control
is similarly dense; Meta-World and manipulation generally get sparser and
stagier (reach, then grasp, then move). For world-model training this
mostly raises the stakes on the reward head: a sparse reward predicted
through a drifting latent is how imagination training silently fails, and
your dream-cheating experiment from research.md becomes directly relevant.

Action spaces. Steer/gas/brake becomes 6 to 30 dimensional joint torques or
delta end-effector poses, still continuous, still Box-shaped. Stage 2's
tanh-squashed Gaussian actor carries over unchanged except for the
dimension. What is genuinely new is action *semantics*: torque control is
sensitive to control frequency, and world models care because the
action-to-effect delay in frames changes with it.

Everything else that quietly mattered: episodes terminate on success or
failure rather than on a timer, cameras are fixed instead of chasing (which
kills the camera-following confound from the memory probe, making layout
memory easier to study here), and simulators are deterministic given state,
so dream-vs-real divergence is attributable in a way real-robot data never
is.

## How stage 2's code ports

The port is smaller than a rewrite, and it should stay one code path:

- Encoder/decoder: swap the 64x64 conv stack input for pixels at the env's
  render size, add a 2-3 layer MLP for the proprio vector, concatenate
  features before the RSSM. Decode both (pixel head and proprio head); the
  proprio reconstruction is nearly free and is a strong training signal.
- RSSM: unchanged. It never knew what the observations meant.
- Reward and continue heads: unchanged in structure, now load-bearing (see
  above and wmgym.md).
- Actor-critic: change the action dimension; keep tanh-Gaussian.
- Collection loop: gymnasium API for dm_control via shimmy, or the suite's
  native API; either way, the same replay format you cached to disk all
  course. On the MJX rung this component gets rebuilt around batched env
  steps, and it is the only one that does.

The honest sequencing: do the dm_control rung as part of the WM-Gym plan
(same port, public scoreboard for free), spend real time on Meta-World
because multi-task world modeling is the open, interesting part, and treat
the MJX/ManiSkill rung as infrastructure practice for the open-dreamer
scale of engineering.
