"""Recurrent State Space Model: the memory of the world model.

State = (h, z). h is deterministic (GRU), z is stochastic (16 categorical
variables with 16 classes each). The prior predicts z from h alone, the
posterior corrects it with the current frame's embedding.
"""

import torch
import torch.distributions as td
import torch.nn as nn
import torch.nn.functional as F

# Weight on the dynamics side of the balanced KL. At 0.8 most of the gradient
# pushes the prior toward the posterior (teach the predictor what perception
# saw) and only 0.2 pulls the posterior toward the prior (keep perception
# predictable). Set to 0.0 the prior never learns and imagination decouples
# from reality.
KL_BALANCE_ALPHA = 0.8

# Free bits, in nats. KL below this threshold is clamped and produces no
# gradient, so the optimizer stops squeezing the latent once its information
# is already cheap and reconstruction keeps the capacity it needs.
FREE_NATS = 1.0

# Mix 1% uniform into every categorical (DreamerV3 "unimix") so no class ever
# hits probability zero, which keeps the KL between prior and posterior finite.
UNIMIX = 0.01


class RSSM(nn.Module):
    def __init__(self, action_dim=3, embed_dim=1024, deter_dim=512,
                 hidden_dim=512, n_vars=16, n_classes=16):
        super().__init__()
        self.n_vars = n_vars
        self.n_classes = n_classes
        self.stoch_dim = n_vars * n_classes
        self.deter_dim = deter_dim
        # Exercises can flip this to feel why stochastic latents matter.
        self.deterministic_z = False

        self.za = nn.Sequential(
            nn.Linear(self.stoch_dim + action_dim, hidden_dim), nn.SiLU())
        self.gru = nn.GRUCell(hidden_dim, deter_dim)
        self.prior_net = nn.Sequential(
            nn.Linear(deter_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, self.stoch_dim))
        self.post_net = nn.Sequential(
            nn.Linear(deter_dim + embed_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, self.stoch_dim))

    def initial(self, batch, device):
        h = torch.zeros(batch, self.deter_dim, device=device)
        z = torch.zeros(batch, self.stoch_dim, device=device)
        return h, z

    def _logits(self, raw):
        logits = raw.view(*raw.shape[:-1], self.n_vars, self.n_classes)
        probs = torch.softmax(logits, -1)
        probs = (1.0 - UNIMIX) * probs + UNIMIX / self.n_classes
        return torch.log(probs)

    def sample(self, logits):
        probs = torch.softmax(logits, -1)
        if self.deterministic_z:
            idx = probs.argmax(-1)
        else:
            idx = td.Categorical(probs=probs).sample()
        onehot = F.one_hot(idx, self.n_classes).float()
        # Straight-through: forward pass uses the hard one-hot sample, backward
        # pass flows through the softmax probabilities.
        z = onehot + probs - probs.detach()
        return z.flatten(-2)

    def _deter_step(self, h, z, action):
        return self.gru(self.za(torch.cat([z, action], -1)), h)

    def img_step(self, h, z, action):
        """One step of pure prediction: no frame, prior only."""
        h = self._deter_step(h, z, action)
        prior_logits = self._logits(self.prior_net(h))
        z = self.sample(prior_logits)
        return h, z, prior_logits

    def obs_step(self, h, z, action, embed):
        """One step of perception: prior for the KL, posterior for the state."""
        h = self._deter_step(h, z, action)
        prior_logits = self._logits(self.prior_net(h))
        post_logits = self._logits(self.post_net(torch.cat([h, embed], -1)))
        z = self.sample(post_logits)
        return h, z, prior_logits, post_logits

    def observe(self, embeds, actions, state=None):
        """Roll the posterior over a subsequence.

        embeds: (B, T, E). actions: (B, T, A), where actions[:, t] is the
        action that led into frame t (zeros at t=0 of an episode).
        """
        B, T, _ = embeds.shape
        if state is None:
            h, z = self.initial(B, embeds.device)
        else:
            h, z = state
        hs, zs, priors, posts = [], [], [], []
        for t in range(T):
            h, z, pl, ql = self.obs_step(h, z, actions[:, t], embeds[:, t])
            hs.append(h)
            zs.append(z)
            priors.append(pl)
            posts.append(ql)
        return (torch.stack(hs, 1), torch.stack(zs, 1),
                torch.stack(priors, 1), torch.stack(posts, 1))


def kl_loss(post_logits, prior_logits, alpha=KL_BALANCE_ALPHA):
    """Balanced KL between posterior and prior with free bits.

    Returns (loss, mean_kl_nats). The KL is summed over the 16 categorical
    variables, so free bits apply to the whole latent, not per variable.
    """
    def kl(p, q):
        p = td.Independent(td.OneHotCategorical(logits=p), 1)
        q = td.Independent(td.OneHotCategorical(logits=q), 1)
        return td.kl_divergence(p, q)

    # dyn trains the prior to match what perception produced.
    dyn = kl(post_logits.detach(), prior_logits)
    # rep keeps the posterior close to what the prior can predict.
    rep = kl(post_logits, prior_logits.detach())
    loss = (alpha * torch.clamp(dyn, min=FREE_NATS)
            + (1.0 - alpha) * torch.clamp(rep, min=FREE_NATS)).mean()
    return loss, dyn.mean().detach()
