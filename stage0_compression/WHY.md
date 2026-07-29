# Stage 0: Compression

## The core idea

A 64x64 RGB frame is 4096 pixels, 3 channels each: 12288 numbers. We are going to force a network to squeeze each frame through a bottleneck of 32 numbers and then rebuild the frame from those 32 alone. That is a compression ratio of 384 to 1. There is no way to do that by memorizing pixels. The only way to survive the bottleneck is to discover what actually varies in the data: where the road bends, where the car sits, how much grass is on each side. Compression is not a preprocessing trick here. It is the mechanism that forces the network to build a description of the world, and that description (the latent vector z) is what every later stage of this course operates on. The dynamics model in the next stage never sees a pixel. It lives entirely in z space.

**Hand-write `models.py` yourself before reading mine.** It is about 100 lines, the architecture is given below, and the reparameterization trick is the one piece of this stage you should feel in your fingers, not just read.

## Design choices and why

**Frames resized to 64x64 uint8.** Same convention for the whole course. 96x96 native CarRacing frames carry no extra information we need, and 64x64 makes the conv arithmetic land exactly on the classic World Models architecture, so you can compare your numbers against the paper.

**Random policy with frame skip for data collection.** We hold each random action for 4 environment steps and store one frame per action. Consecutive raw frames are near-identical; without the skip, a 20k frame dataset is really a 5k frame dataset with each frame counted four times, and the model gets an easy, misleading loss. We also throw away the first 30 steps of each episode because CarRacing opens with a zoom-in animation that is not real driving data. Collection is seeded, so the dataset is reproducible byte for byte.

**Two models, AE then VAE.** The plain autoencoder (ConvAE) maps a frame to a single point z and back. It learns compression fine, and its reconstructions are usually a bit sharper than the VAE's. So why bother with the VAE? Because of what lives *between* the points. An AE is free to scatter its codes anywhere in R^32, with voids between them where the decoder was never trained. Decode a point in one of those holes and you get garbage. The VAE instead encodes each frame as a distribution, a Gaussian with mean mu and variance exp(logvar), and trains the decoder on *samples* from that distribution. Every training step the decoder sees a slightly different z for the same frame, so it is forced to produce sensible output over a whole neighborhood, not a single point. Neighborhoods overlap, the space fills in, and the map from z to images becomes smooth. This matters enormously later: in the dream stages we will roll a dynamics model forward in latent space, and its predictions will land between and around the codes of real frames. You cannot dream in a latent space with holes.

**The KL term.** The VAE loss adds KL(q(z|x) || N(0, I)): the penalty for the encoder's Gaussians straying from a unit Gaussian at the origin. Read it as a zoning law. Reconstruction pressure wants every frame to have its own private, far-flung code; the KL term pulls all codes toward the origin and keeps their variances near 1, so the whole dataset occupies one connected, organized region. Sampling z ~ N(0, I) then lands you inside the trained region, which is exactly what generation and dreaming require. Beta scales this term: the default 1.0 is the standard ELBO, lower trades organization for sharpness, higher trades sharpness for organization until there is nothing left (see below).

**The reparameterization trick.** We need gradients to flow from the reconstruction loss back through the sampling step into mu and logvar. "Draw z from N(mu, sigma^2)" is not differentiable as written, but z = mu + sigma * eps with eps ~ N(0, 1) is the same distribution with the randomness moved into an input. Now z is a deterministic, differentiable function of mu and logvar. One line of code, and it is the line that makes VAEs trainable. There is a test in this stage that checks the gradient actually reaches fc_logvar; if you hand-write models.py, make sure that test passes against your version.

**Sum-based losses.** Reconstruction is summed squared error per image, KL is summed per image, both averaged over the batch. If you instead take the mean over 12288 pixels, the reconstruction term shrinks by four orders of magnitude relative to the KL term and beta=1.0 silently becomes beta=12288 in disguise. Getting this scaling wrong is the single most common way people accidentally collapse their VAE.

**Why reconstructions are blurry.** MSE is minimized by the mean of all plausible outputs. Given 32 numbers, many detailed frames are consistent with them: the grass texture could be shifted a pixel any direction, the red-white curb stripes could start at any phase. The decoder cannot know which, so the loss-optimal move is to output the average of all of them, and the average of many sharp textures is a blur. This is a property of the loss, not a bug in your code. Road geometry and car position survive because they are exactly the things the 32 numbers do pin down.

**Posterior collapse, in one paragraph.** If the KL pressure is too strong relative to reconstruction (large beta, or the scaling bug above), the encoder's cheapest move is to stop encoding: output mu near 0 and logvar near 0 for every frame, making KL almost exactly zero. The decoder, receiving pure noise, learns to output the average frame of the dataset regardless of input. The failure is quiet: the loss still goes down, training looks healthy, but the KL term sitting near zero is the tell, and every reconstruction is the same blurry mean image. A KL of zero is never good news. It means the latent code carries no information at all.

**Why 32 dimensions.** Ha and Schmidhuber used 32 for this exact environment and it is a sensible middle. Too few dims (try 2 in the exercises) and the bottleneck cannot hold road curvature, car pose, and track layout at once, so reconstructions degrade visibly. Too many and the model stops being forced to find structure: it can afford to spend dims on texture noise, and the latent space becomes a worse substrate for the dynamics model, which has to predict every one of those dims. 32 is small enough to force abstraction, large enough not to cripple reconstruction, and keeping Ha's number means our later stages stay comparable to the paper.

**Architecture specifics.** Four stride-2 convs (32, 64, 128, 256 channels, 4x4 kernels) take 64x64 down to 2x2x256 = 1024 features; the decoder mirrors this with four transposed convs (kernel sizes 5, 5, 6, 6) that land exactly back on 64x64. Sigmoid output because pixels are normalized to [0, 1]. This is the World Models encoder/decoder nearly verbatim, on purpose.

## Common failure modes

- KL near zero from step one and flat: posterior collapse. Check beta and check that your reconstruction loss sums over pixels rather than averaging.
- Loss decreases but reconstructions are a uniform green-brown smear: same collapse, seen from the pixel side.
- Reconstructions fine but interpolations between two frames pass through garbage: your latent space has holes; this is the expected AE behavior and the reason the VAE exists.
- Loss explodes or NaNs: logvar unbounded is the usual suspect; lower the learning rate first.
- Suspiciously easy loss: consecutive frames nearly identical because frame skip was removed; the model is being graded on duplicates.

## Backward and forward

Backward: nothing, this is the start. Forward: everything. Stage 1 trains a dynamics model that takes (z_t, action) and predicts z_{t+1}, entirely inside the 32-dim space built here; the frozen VAE from this stage is its eyes. The dream stages then roll that dynamics model forward without touching the environment, decoding latent trajectories back to pixels only to look at them. Every later stage defines its own copy of this VAE so it stays self-contained, and each copy mirrors the one you build here.

## You get it when

1. Why can a network that reconstructs frames through a 32-number bottleneck be said to "understand" the scene, in what limited sense?
2. What can go wrong if you train a dynamics model in the latent space of a plain AE instead of a VAE?
3. What two things does the KL term push the encoder's mu and logvar toward, and why does that make z ~ N(0, I) a useful sampling distribution?
4. Why does z = mu + sigma * eps make sampling differentiable when "sample from N(mu, sigma^2)" is not?
5. Your KL term reads 0.02 after thousands of steps. Is that good compression or a disaster, and what would the reconstructions look like?
6. Why does an MSE-trained decoder blur grass texture but keep road edges relatively sharp?
7. If you average the reconstruction loss per pixel instead of summing, what does beta=1.0 effectively become, and what happens to training?
8. What breaks first when you drop latent dim from 32 to 2, and what would you lose by raising it to 512?
