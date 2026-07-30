"""MDN-RNN: an LSTM over (z_t, a_t) with a mixture density head on z_{t+1}.

The head outputs, for every latent dimension independently, a mixture of
n_gaussians univariate gaussians: mixture logits, means, and log stds.
Training minimizes the negative log likelihood of the true next latent.

Temperature at sampling time: mixture logits are scaled by 1/T and each
gaussian std by sqrt(T). T < 1 makes the dream more deterministic than the
model believes, T > 1 makes it noisier.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

LOG_2PI = math.log(2.0 * math.pi)


class MDNRNN(nn.Module):
    def __init__(self, z_dim: int = 32, action_dim: int = 3,
                 hidden_dim: int = 256, n_gaussians: int = 5):
        super().__init__()
        self.z_dim = z_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.n_gaussians = n_gaussians
        self.lstm = nn.LSTM(z_dim + action_dim, hidden_dim, batch_first=True)
        self.head = nn.Linear(hidden_dim, 3 * n_gaussians * z_dim)

    def forward(self, z, a, hidden=None):
        """z: (B, T, z_dim), a: (B, T, action_dim).

        Returns ((logpi, mu, logstd), hidden), each param shaped
        (B, T, z_dim, n_gaussians). logpi is already log-normalized.
        """
        out, hidden = self.lstm(torch.cat([z, a], dim=-1), hidden)
        B, T, _ = out.shape
        p = self.head(out).view(B, T, self.z_dim, self.n_gaussians, 3)
        logpi = F.log_softmax(p[..., 0], dim=-1)
        mu = p[..., 1]
        # Clamp keeps exp(logstd) finite early in training.
        logstd = p[..., 2].clamp(-6.0, 3.0)
        return (logpi, mu, logstd), hidden


def mdn_nll(logpi, mu, logstd, z_next):
    """Negative log likelihood of z_next under the per-dim mixtures.

    z_next: (B, T, z_dim). Params: (B, T, z_dim, n_gaussians).
    The joint log likelihood of a latent vector sums over its independent
    dimensions; the result is then averaged over batch and time.
    """
    target = z_next.unsqueeze(-1)
    log_prob = -0.5 * ((target - mu) / logstd.exp()) ** 2 - logstd - 0.5 * LOG_2PI
    log_mix = torch.logsumexp(logpi + log_prob, dim=-1)
    return -log_mix.sum(dim=-1).mean()


def mdn_sample(logpi, mu, logstd, temperature: float = 1.0, generator=None):
    """Sample one z from mixture params of shape (..., z_dim, n_gaussians).

    Returns a tensor shaped (..., z_dim).
    """
    k = logpi.shape[-1]
    pi = torch.softmax(logpi / temperature, dim=-1)
    idx = torch.multinomial(pi.reshape(-1, k), 1, generator=generator)
    idx = idx.view(*pi.shape[:-1], 1)
    mu_sel = mu.gather(-1, idx).squeeze(-1)
    std_sel = logstd.gather(-1, idx).squeeze(-1).exp() * math.sqrt(temperature)
    eps = torch.empty_like(mu_sel).normal_(generator=generator)
    return mu_sel + std_sel * eps
