# Stage 3: the world as tokens

## The core idea

Turn each video frame into 64 discrete symbols from a fixed vocabulary of 256, then treat "predict the future" as "predict the next token", exactly the way a language model predicts the next word. A VQ-VAE learns the vocabulary (a codebook of visual snippets), and a small GPT learns the grammar of the world: which token grids follow which, given the action taken. This is the IRIS recipe, and it is still the skeleton inside Genie-class models.

Why bother making the world discrete? Three reasons.

1. It turns video into a language problem. Once frames are token sequences, the model is a plain categorical next-token predictor. Cross entropy over 256 classes is an exact likelihood, not a Gaussian approximation of pixels. There is no blurriness from averaging futures: when the model is unsure whether the car turns left or right, it puts probability on both token patterns and sampling picks one, instead of regressing to the gray mush between them.
2. You inherit the entire language-model toolbox for free: temperature and nucleus sampling, KV caches, beam search if you want it, known scaling laws, known training recipes, known infrastructure. Nothing in this stage's transformer knows it is looking at a racetrack.
3. Discreteness is a strong regularizer. The dynamics model cannot exploit tiny continuous perturbations in the latent; it must commit to one of 256 options per cell, which keeps rollouts from sliding off the data manifold in the smooth, hard-to-notice way stage 2 latents do.

The price is a quantization floor (the decoder can only paint what the codebook can express) and slow sampling (64 transformer calls per frame). Stage 4 addresses both.

## Design choices and why

**64x64 frames, 8x8 token grid, codebook 256 x 64.** Three stride-2 convs take 64 to 8. An 8x8 grid at vocab 256 means each frame is 64 bytes: a 192x compression of the 12288-byte frame, and short enough that a several-frame window fits in a small transformer. 256 codes is deliberately modest; CarRacing has little visual diversity (road, grass, curb, car, HUD bar), and a small vocabulary makes codebook health easy to inspect.

**Straight-through estimator, in one paragraph.** The argmin that snaps an encoder vector to its nearest code has zero gradient almost everywhere, so backprop through it would give the encoder nothing. The fix: in the forward pass use the quantized vector `z_q`, but write it as `z_e + (z_q - z_e).detach()`. The detached term is constant to autograd, so the gradient of the reconstruction loss with respect to this expression is exactly its gradient with respect to `z_e`. In other words, we copy the decoder's gradient across the non-differentiable pick, pretending quantization was the identity. It is a biased estimator, and the bias is small only while `z_e` stays close to its assigned code, which is precisely what the commitment loss (beta 0.25) enforces by pulling `z_e` toward the chosen, detached codebook entry.

**EMA codebook updates instead of a codebook gradient loss.** The codebook entries are updated as exponential moving averages of the encoder vectors assigned to them (a soft online k-means), not by gradient descent. EMA converges faster, has no learning-rate coupling with the encoder, and is what most modern VQ implementations use. Note the codebook is a buffer, not a parameter: only the encoder and decoder are in the optimizer.

**Codebook collapse and the standard fixes.** The classic failure: a few codes get picked early, get updated, get even more attractive, and the other 250 never move again. Reconstructions turn into a handful of repeated textures and perplexity craters to single digits. Two fixes, both in `vqvae.py`: EMA with Laplace smoothing keeps rare codes from being divided to zero, and dead-code reinit teleports any code whose EMA usage falls below a threshold onto a random encoder output from the current batch, putting it back where the data actually lives. The `usage_histogram()` helper exists so you can watch this instead of taking it on faith; the first break-it lab turns the fixes off.

**One action token per frame.** The sequence is `[a_0, z_0 (64 tokens), a_1, z_1 (64 tokens), ...]`, where `a_t` is the action that produced frame `t` (so `a_0` is zeros), embedded by a linear map from the continuous (steer, gas, brake) vector into model width. One token is enough because the action is 3 numbers, not an image; giving it 64 tokens would waste sequence budget. Putting it before its frame means every frame token is generated already knowing what the driver did, which is what makes the model conditional (a controllable world) rather than a passive video predictor. The loss is masked at action positions: actions come from the controller and are never predicted.

**Learned positional embeddings.** RoPE would work too; for a fixed, short block size, a learned table is fewer moving parts and makes the KV-cache offset logic trivial (position id = cache length).

**Teacher forcing, free running, and exposure bias.** Training always shows the model a real prefix and asks for the next real token: teacher forcing. Rollout feeds the model its own sampled tokens: free running. The model has never seen its own slightly-wrong outputs as input, so a small error puts it in a state distribution it was never trained on, its next prediction is a bit worse, and errors compound frame over frame. That compounding is exposure bias, and the drift curve in `rollout3.py` (PSNR against the ground-truth continuation, per horizon step) is a direct measurement of it. Expect high PSNR at horizon 1 and a bend downward within a few frames.

**KV cache.** Generating one frame is 64 sequential forward passes. Without a cache each pass would recompute attention over the whole prefix, making frame `t` cost O(t^2 x 65^2) attention work. The cache stores per-layer keys and values so each new token attends to stored history at the cost of one row. Same trick, same code shape, as every language-model inference stack.

## Common failure modes

- Codebook collapse (see above): watch perplexity and the usage histogram during VQ training, not just the loss.
- Index shuffling: if you flatten the 8x8 grid in a different order in the encoder and the decoder (or the rollout), everything trains fine and rollouts are scrambled tiles. Keep row-major everywhere.
- Loss on action positions: if you forget to mask them, the model wastes capacity trying to predict the controller and the token loss plateaus high.
- Off-by-one in the interleave: position `p` predicts element `p+1`. The action token predicts the first frame token; the last frame token of frame `t` would predict `a_{t+1}` and must be ignored.
- Sampling at temperature 0: deterministic rollouts lock onto the mode and freeze or loop (break-it lab 2).

## Backward and forward

Backward: stage 0 compressed frames into a continuous latent with a VAE, and stages 1 and 2 rolled latents forward with recurrent networks. Two inherited limits motivate this stage. Regressing a continuous next latent fights blur, because a mean-seeking loss averages possible futures (stage 2's categorical z softened this, but stages 0 and 1 live with it fully). And stage 2 ended on a structural bottleneck: its GRU processes time strictly in sequence, so training cannot parallelize across the sequence, and its fixed-size h must remember everything. Discrete tokens fix the first (classification has an exact likelihood and no averaging), and the transformer fixes the second (parallel training over the whole window, memory that attends instead of compresses), trading blur for a quantization floor.

Forward: stage 4 keeps this exact architecture (tokenizer plus autoregressive transformer over interleaved actions and latents) but swaps exact tokens for continuous latents with noise-tolerant training: instead of demanding the exact next token, the model learns to denoise or predict targets that survive small perturbations, which is how you get past the 64-calls-per-frame sampling cost and the codebook ceiling. When you read a Genie-style paper, you should recognize this stage's diagram inside it: a spatial tokenizer, one action embedding per frame, a causal transformer over the interleaved stream.

## Write it yourself first

Before reading `vqvae.py`, write the quantizer yourself: nearest-neighbor lookup, straight-through trick, commitment loss, EMA update, dead-code reinit. It is under 60 lines and contains every idea in this stage that is not already a standard transformer.

## You get it when

1. Why does cross entropy over tokens give sharp rollouts where MSE on pixels or latents gives blur?
2. In `z_e + (z_q - z_e).detach()`, what does the decoder see in the forward pass, and what gradient does the encoder receive in the backward pass?
3. The commitment loss pulls `z_e` toward `sg[z_q]`. What updates the codebook, and why is it not in the optimizer?
4. Your codebook perplexity drops from 90 to 6 during training. What happened mechanically, and which two mechanisms in the quantizer are supposed to prevent it?
5. Why is the action one token, and why does it sit before its frame instead of after?
6. Which sequence positions are excluded from the loss, and what goes wrong if they are not?
7. The model is near-perfect at next-token prediction on held-out data, yet 20-frame rollouts fall apart. What is the name of this gap and what causes it?
8. During rollout, why does generating frame 10 cost the same per token as generating frame 2, and what data structure makes that true?
