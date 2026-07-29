# Stage 2: Dreamer, or learning to act inside your own prediction

## The core idea

Stage 1 already showed the trick: learn a model of the world, then train the
controller inside the model instead of the real environment. This stage keeps
that idea and replaces every crude part with better machinery. The model gets
a recurrent state so it remembers the past. The controller becomes an actor
trained by gradient descent through the model's own predictions, not by
evolution. The result is DreamerV3, simplified: collect a little real
experience, train the world model on it, dream 15 steps ahead thousands of
times in parallel, and push the policy uphill along the gradient of imagined
return.

If you write one file yourself before reading mine, write `rssm.py`. It is
about a hundred lines and it contains the entire conceptual load of the
stage: the split between deterministic and stochastic state, the split
between prior and posterior, and the KL that welds them together. Everything
else is heads and plumbing.

## Why a recurrent state instead of a frame stack

A frame stack answers "what happened in the last k frames" by paying for k
frames of convolution at every step. A recurrent state answers "what do I
need to remember about everything so far" and pays for one frame of
convolution plus one GRU update. Three costs favor the recurrent state.
Memory: velocity, or whether you clipped grass 40 steps ago, does not fit in
a 4-frame window unless you make the window huge. Partial observability: a
single CarRacing frame does not tell you your speed or which way the track
curves beyond the horizon; the state can carry an estimate of both. Cost per
imagined step: this is the decisive one. During dreaming there are no frames
at all. If your state were a frame stack you would have to decode pixels just
to re-encode them. With (h, z) an imagined step is a GRU update and an MLP,
so you can afford thousands of parallel rollouts per gradient step.

## Posterior versus prior: perception versus prediction

The RSSM has two ways to produce the stochastic latent z. The posterior looks
at h and the encoding of the current frame: that is perception, "given what I
see, what state am I in". The prior looks at h alone: that is prediction,
"given what I remember, what state should I be in before looking". Training
uses the posterior; dreaming can only use the prior, because in a dream there
is no frame to look at.

The KL between them is what makes dreaming trustworthy. It plays both
directions at once. Pushing the prior toward the posterior trains the
predictor to match what perception actually saw. Pushing the posterior toward
the prior keeps perception from encoding things the predictor could never
anticipate, because a latent the prior cannot predict is useless in a dream.
These two pressures should not be equal. If the posterior bends too easily
toward a bad prior, perception degrades to protect a predictor that has not
earned it. So the loss is balanced: `KL_BALANCE_ALPHA = 0.8` of the gradient
goes into the prior side (posterior detached) and 0.2 into the posterior side
(prior detached). The predictor chases perception hard; perception yields
only a little.

## Why categorical latents instead of Gaussians

The original world models line, and DreamerV1, used Gaussian latents. V2 and
V3 found that a vector of categoricals (here 16 variables with 16 classes
each) simply works better on these tasks, and the finding has been stable.
Three reasons to like them. The straight-through estimator (use the hard
one-hot sample forward, route the gradient through the softmax backward) is
biased but simple and remarkably robust; there is no reparameterization
noise scale to get wrong. A categorical cannot posterior-collapse the way a
Gaussian can, where the variance quietly inflates until the latent carries
nothing and the KL is zero. And the KL between two categoricals is exact,
bounded, and well behaved; no exploding log-variances. A 1% uniform mixture
(`UNIMIX`) keeps every class at nonzero probability so the KL stays finite.

## Free bits

`FREE_NATS = 1.0`: any KL below one nat is clamped and produces no gradient.
Without this, the optimizer keeps squeezing the latent even when the
information it carries is already cheap, because reducing KL is the easiest
loss term to improve. The squeeze starves reconstruction, which starves the
reward head, which starves the policy. Free bits say: below one nat, the
information is paid for, stop optimizing it and spend gradient elsewhere. You
can watch this in the smoke run: the KL sits far below 1 nat, and because of
free bits the KL term contributes exactly zero gradient there.

## Why imagine only 15 steps

Two clocks run against you in a dream. Model error compounds: each imagined
step feeds the model its own output, so a 2% per-step error is noise at step
5 and fiction at step 50. And credit assignment stretches thin: the gradient
of the return at step 40 with respect to the action at step 1 passes through
40 GRU updates and is mush by the time it arrives. H = 15 is short enough
that the model is still mostly honest and gradients still mean something,
and long enough to see a curve coming. The lambda return then fills in for
everything past the horizon through the critic's bootstrap.

## Lambda returns, honestly

You want to score each imagined state. Two pure options exist. Monte Carlo:
sum the imagined rewards to the horizon; unbiased with respect to the model
but high variance, and blind past step 15. Pure bootstrap: one reward plus
the critic's estimate of the next state; low variance but inherits every bias
the critic has. The lambda return blends every n-step estimate in between
with geometric weights: with lambda = 0.95, short backups get a little
weight, long backups get more, and the recursion
`R_t = r_t + gamma * ((1 - lambda) * v_t + lambda * R_{t+1})` computes the
whole blend in one backward sweep. It is a variance-bias dial, nothing more.
We set it near 1 because imagined rollouts are cheap and the critic is the
weakest link early in training, so we lean toward real (imagined) rewards
and use the critic mostly to summarize the world past the horizon.

## Symlog

CarRacing rewards are small and frequent (about +3 per tile, a constant
-0.1 per step), but other environments pay 10,000 at once, and even here the
returns the critic must represent span two orders of magnitude during
training. A squared error on raw values makes the largest scale dominate the
gradient. symlog(x) = sign(x) * log(1 + |x|) compresses large magnitudes,
leaves small ones almost untouched, is exactly invertible by symexp, and
costs nothing. Reward head and critic both predict in symlog space and
readers apply symexp. It is the cheapest robustness trick in the whole
recipe.

## Why backprop through the dream replaces evolution

Stage 1 treated the controller as a black box and asked evolution to wiggle
its weights. That works when the controller is a few hundred parameters, and
it needs nothing differentiable. But the world model we just built is
differentiable end to end: state through action through next state through
reward. Backpropagating the imagined return through that chain gives every
actor parameter its own exact-ish gradient in one pass. The comparison is
lopsided: evolution needs many full rollouts to estimate one noisy direction
in weight space, and the noise grows with parameter count; backprop gets a
per-parameter signal from the same rollouts and scales to millions of
parameters without changing anything. Sample efficiency is the visible win;
scaling with parameter count is the structural one. The price is that the
gradient is only as good as the model, which is exactly why the KL balance,
free bits, and the short horizon exist: they keep the dream honest enough to
differentiate through.

## Failure modes to recognize

KL collapse: kl pinned near zero while reconstruction stays bad means
information has stopped flowing into z; check the gradient path first (the
straight-through lab shows exactly this signature). Crippled perception:
reconstruction degrading while kl also drops can mean the posterior is being
over-regularized toward a prior that has not earned it (the alpha = 0 lab);
dreams then come from an untrained predictor and the actor optimizes
fiction. Value explosion: img_return growing
while real returns do not means the actor found a bug in the model, usually
reward head extrapolation; shorter horizon and more world model training per
actor step help. Dead gradient through tanh: actor std collapsing to the
floor with entropy near its minimum means the entropy bonus is too small.

## Backward and forward

Backward: stage 0 built the compressor (our encoder and decoder are the same
conv trunk, minus the VAE head), and stage 1 built the first dream loop with
an evolved controller. This stage is the same loop with memory, a trained
predictor, and gradients. Forward: the RSSM is now the bottleneck. Its GRU
processes time strictly sequentially, so training cannot parallelize across
the sequence, and its fixed-size h must remember everything. The next step
in this line replaces the recurrence with a transformer over latent tokens,
which trades the O(1) recurrent step for parallel training and a memory that
attends instead of compresses. When you meet that architecture, everything
in this stage except the GRU survives.

## You get it when

1. In a dream there are no frames. Which of prior and posterior can you use,
   and why does that force the KL to exist?
2. Why does alpha = 0.8 push the prior toward the posterior harder than the
   reverse, and what breaks if you flip it?
3. Your KL reads 0.4 nats with FREE_NATS = 1.0. How much gradient is the KL
   term contributing right now?
4. Why is one imagined step cheaper than one real step by orders of
   magnitude, and which design choice makes that true?
5. What two costs grow as you push the imagination horizon from 15 to 50,
   and which one the lambda return can compensate for?
6. In R_t = r + gamma * ((1 - lambda) * v + lambda * R_{t+1}), what do you
   recover at lambda = 0 and at lambda = 1?
7. Why can a Gaussian latent collapse while a categorical with unimix
   cannot?
8. The actor never sees a single real transition during its update. Where
   does its gradient physically come from?
