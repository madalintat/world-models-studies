# Stage 1: Ha and Schmidhuber's world model (V + M + C)

Reference: Ha and Schmidhuber, "World Models", 2018. Read the interactive
version at worldmodels.github.io after this file.

**Write `mdnrnn.py` yourself before opening the version here.** You know the
LSTM part already. The part worth sweating over is the mixture density head
and its negative log likelihood: get the logsumexp right, get the per-dimension
mixture right, and you understand the whole model.

## The core idea

Stage 0 taught a VAE to squash a 12288-number frame into 32 numbers. This
stage adds the two missing pieces: a model that predicts how those 32 numbers
evolve when you act (M, an MDN-RNN), and a tiny policy that reads the
compressed state and the predictor's memory to pick actions (C, one linear
layer). Once M exists, you can unplug the simulator entirely: start from a
real latent, let C pick actions, let M imagine the consequences, repeat. The
agent trains inside its own hallucination. That is the whole field of world
models in one sentence, and this 2018 paper is its cleanest statement.

## Design choices and why

**Predict latents, not pixels.** Two reasons. Cost: one dream step is an LSTM
step on a 35-dim input instead of a conv net on 12288 pixels, so a thousand
imagined steps cost milliseconds and the controller can be evaluated thousands
of times per minute on a CPU. Loss behavior: pixel-space MSE punishes any
sharp guess that is slightly misplaced more than it punishes a blurry average,
so pixel predictors converge to mush. In latent space the VAE has already paid
the blur tax once; the dynamics model only has to move 32 numbers around.

**Why a mixture (MDN) and not a plain gaussian head.** The future is
multimodal. Approaching a corner, the next latent depends on whether the track
curves left or right, and on noise you cannot see. A single gaussian must put
its mean somewhere, so it averages incompatible futures into a point between
them, which often corresponds to no valid frame at all (a road that goes both
ways at once, decoded as gray smear). Five gaussians per latent dimension let
the model say "either this or that", and sampling picks one coherent branch.
We use per-dimension mixtures exactly as the original implementation did:
the head outputs 5 (logit, mean, logstd) triples for each of the 32
dimensions, and the NLL is a logsumexp over the 5 components, summed over
dimensions and averaged over time and batch.

**Why train the controller in the dream.** Speed: the dream runs orders of
magnitude faster than Box2D and never needs rendering. Safety: a dream crash
costs nothing, which matters a lot once the "environment" is a physical robot.
And it is the intellectual point of the exercise: if the learned model is good
enough to train against, the real environment is only needed to collect data
and to grade the final answer. Honesty note: for CarRacing the original paper
actually trained C in the real environment and used dream training for the
VizDoom experiment. We train in the dream anyway because experiencing the
dream-reality gap firsthand is the lesson, and we provide `--real` for the
faithful variant.

**The proxy reward, honestly.** Our M predicts only the next z, not reward
(the real CarRacing reward counts visited track tiles, which lives in the
simulator, not in our latents). So the dream needs a stand-in objective. We
use: gas minus brake, minus 0.1 times absolute steering, minus 0.05 times
mean(z squared). The first terms say "move and do not thrash". The last term
says "keep the dream on-distribution", since latents near the VAE prior have
mean square around 1 and off-manifold dreams drift to large norms. This is a
crude proxy and we say so: its optimum (throttle pinned) is wrong in corners,
and the exercises measure exactly how much real score that costs. The clean
fix, predicting reward alongside z, is what Dreamer does in the next stage.

**Temperature: the first anti-cheating tool.** At sampling time we scale the
mixture logits by 1/T and each gaussian std by sqrt(T). T near 0 collapses the
dream to the most likely branch with almost no noise: a smooth, deterministic
movie that is easy to control and easy to exploit, because any systematic
error in M becomes a reliable lever the controller can lean on. T above 1
injects extra noise everywhere: strategies that only work in one exact
imagined future stop paying off, so the controller is pushed toward policies
robust to model error. Hotter dream, harder to exploit. The paper found
T = 1.15 dreams transferred to real VizDoom better than T = 1.0 ones. This is
the ancestor of every later trick against world-model exploitation.

**Why CMA-ES and not gradients.** The controller has exactly
(32 + 256 + 1) * 3 = 867 parameters. Backpropagating a return through
hundreds of dream steps would mean differentiating through categorical
mixture sampling (painful) and through a long recurrent chain (unstable).
With 867 parameters you do not need any of that: evolution strategies treat
the whole rollout as a black box, and CMA-ES in particular adapts a full
covariance over parameter space, which is affordable at this dimension. Ask
for a population, evaluate returns, update mean and covariance, repeat. Our
`cmaes.py` is about 100 lines of numpy following Hansen's tutorial, and it is
the same algorithm class the paper used.

**The honest gap.** The number the dream reports is not the number the world
pays. The gap has three sources stacked on top of each other: the VAE throws
away information the task needs (exact curb position), M's dynamics are
wrong in ways C will find and exploit, and here our proxy reward is not the
real reward at all. Expect dream returns to climb generation after generation
while real returns lag far behind. Measuring that gap, rather than pretending
it away, is the mature move; closing it is what the rest of the course is
about.

## Common failure modes

- MDN NLL goes NaN: exploding stds early in training. We clamp logstd and
  clip gradients; if you removed those, that is why.
- All mixture weights collapse onto one component: not necessarily a bug.
  On easy stretches one gaussian suffices; check pi entropy near corners.
- Dream diverges into abstract art after 50 steps: your M was trained on
  too little data or too short sequences; the controller is now being
  optimized inside noise.
- CMA-ES fitness improves but the video shows the car doing nothing: read
  your proxy reward again and check sign conventions (CMA-ES here minimizes,
  so fitness is negative return).
- Real evaluation stuck at low score with full throttle: that is the proxy
  reward doing exactly what you asked, not a bug. See the cheating lab.

## Backward and forward

Backward: V is stage 0's VAE, copied here unchanged so the stage is
self-contained. Everything stage 0 said about blur and latent geometry now
has consequences: whatever V cannot represent, M cannot predict and C cannot
use. Forward: stage 2 (Dreamer) fixes the three weakest joints you will feel
here: it predicts reward so the proxy hack disappears, it trains V and M
jointly so the latent is shaped for prediction rather than reconstruction,
and it replaces evolution with gradients through the latent dynamics. Stage 3
replaces the whole continuous latent story with discrete tokens and a
transformer.

## You get it when

1. Why does one dream step cost an LSTM step instead of a conv forward pass,
   and why does that matter for how C is trained?
2. A single-gaussian RNN head predicts the latent at a fork in the road.
   What does its mean decode to, and why is that worse than sampling from
   a mixture?
3. In the MDN NLL, why is it logsumexp(logpi + logN) rather than
   sum(pi * logN)?
4. What exactly does temperature multiply, where, and why does T = 0.1 make
   the dream easier for the controller to exploit?
5. Why is CMA-ES a reasonable optimizer at 867 parameters but a bad one at
   867,000?
6. Our dream reward is not the CarRacing reward. What would M have to
   predict for dream training to optimize the real objective, and which
   stage of this course does that?
7. Name the three stacked reasons a dream-trained controller scores worse in
   the real env than in the dream.
8. Where does h (the LSTM hidden state) enter the controller, and what
   information does it carry that z alone does not?
