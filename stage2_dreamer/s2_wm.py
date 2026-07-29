"""World model: encoder, RSSM, decoder, reward head, continue head."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from stage2_dreamer.rssm import RSSM, kl_loss


def symlog(x):
    return torch.sign(x) * torch.log1p(torch.abs(x))


def symexp(x):
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1.0)


class Encoder(nn.Module):
    """(B, 3, 64, 64) in [-0.5, 0.5] to (B, embed_dim)."""

    def __init__(self, depth=32, embed_dim=1024):
        super().__init__()
        d = depth
        self.convs = nn.Sequential(
            nn.Conv2d(3, d, 4, 2, 1), nn.SiLU(),
            nn.Conv2d(d, 2 * d, 4, 2, 1), nn.SiLU(),
            nn.Conv2d(2 * d, 4 * d, 4, 2, 1), nn.SiLU(),
            nn.Conv2d(4 * d, 8 * d, 4, 2, 1), nn.SiLU(),
        )
        self.out = nn.Linear(8 * d * 4 * 4, embed_dim)

    def forward(self, x):
        return self.out(self.convs(x).flatten(1))


class Decoder(nn.Module):
    """(B, feat_dim) to (B, 3, 64, 64) prediction of the preprocessed frame."""

    def __init__(self, feat_dim, depth=32):
        super().__init__()
        d = depth
        self.depth = d
        self.inp = nn.Linear(feat_dim, 8 * d * 4 * 4)
        self.deconvs = nn.Sequential(
            nn.ConvTranspose2d(8 * d, 4 * d, 4, 2, 1), nn.SiLU(),
            nn.ConvTranspose2d(4 * d, 2 * d, 4, 2, 1), nn.SiLU(),
            nn.ConvTranspose2d(2 * d, d, 4, 2, 1), nn.SiLU(),
            nn.ConvTranspose2d(d, 3, 4, 2, 1),
        )

    def forward(self, feat):
        x = self.inp(feat).view(-1, 8 * self.depth, 4, 4)
        return self.deconvs(x)


class ScalarHead(nn.Module):
    def __init__(self, feat_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, feat):
        return self.net(feat).squeeze(-1)


class WorldModel(nn.Module):
    def __init__(self, action_dim=3, depth=32, embed_dim=1024, deter_dim=512,
                 hidden_dim=512, n_vars=16, n_classes=16):
        super().__init__()
        self.rssm = RSSM(action_dim, embed_dim, deter_dim, hidden_dim,
                         n_vars, n_classes)
        self.feat_dim = deter_dim + n_vars * n_classes
        self.encoder = Encoder(depth, embed_dim)
        self.decoder = Decoder(self.feat_dim, depth)
        # Reward head is trained in symlog space; callers apply symexp.
        self.reward_head = ScalarHead(self.feat_dim, hidden_dim)
        # Continue head outputs a logit for P(episode continues).
        self.cont_head = ScalarHead(self.feat_dim, hidden_dim)

    @staticmethod
    def preprocess(obs):
        """uint8 (..., 64, 64, 3) to float (..., 3, 64, 64) in [-0.5, 0.5]."""
        return obs.float().div(255.0).sub(0.5).movedim(-1, -3)

    def loss(self, batch, kl_alpha=None):
        """batch: obs uint8 (B,T,64,64,3), action (B,T,A), reward (B,T),
        cont (B,T). Returns (total_loss, metrics, posterior_states)."""
        obs, action = batch["obs"], batch["action"]
        B, T = obs.shape[:2]
        x = self.preprocess(obs)
        embed = self.encoder(x.flatten(0, 1)).view(B, T, -1)
        hs, zs, prior_l, post_l = self.rssm.observe(embed, action)
        feat = torch.cat([hs, zs], -1)

        recon = self.decoder(feat.flatten(0, 1)).view(B, T, 3, 64, 64)
        # Sum over pixels, mean over batch and time: the reconstruction term
        # must outweigh a KL of a few nats or the latent goes unused.
        recon_loss = ((recon - x) ** 2).sum((-3, -2, -1)).mean()
        reward_loss = ((self.reward_head(feat)
                        - symlog(batch["reward"])) ** 2).mean()
        cont_loss = F.binary_cross_entropy_with_logits(
            self.cont_head(feat), batch["cont"])
        if kl_alpha is None:
            kl, kl_value = kl_loss(post_l, prior_l)
        else:
            kl, kl_value = kl_loss(post_l, prior_l, alpha=kl_alpha)

        total = recon_loss + reward_loss + cont_loss + kl
        metrics = {
            "wm/total": total.item(),
            "wm/recon": recon_loss.item(),
            "wm/reward": reward_loss.item(),
            "wm/cont": cont_loss.item(),
            "wm/kl": kl_value.item(),
        }
        return total, metrics, (hs.detach(), zs.detach())
