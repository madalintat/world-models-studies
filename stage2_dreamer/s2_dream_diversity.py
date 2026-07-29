"""Lab helper: are two dreams from the same state the same dream?

    uv run python -m stage2_dreamer.s2_dream_diversity
"""

import torch

from stage2_dreamer.s2_wm import WorldModel


def main():
    torch.manual_seed(0)
    wm = WorldModel(depth=8, embed_dim=128, deter_dim=64, hidden_dim=64)
    h, z = wm.rssm.initial(1, "cpu")
    action = torch.zeros(1, 3)
    for det in (False, True):
        wm.rssm.deterministic_z = det
        outs = [wm.rssm.img_step(h, z, action)[1] for _ in range(2)]
        label = "deterministic z" if det else "stochastic z   "
        print(f"{label} identical imagined step: "
              f"{torch.equal(outs[0], outs[1])}")


if __name__ == "__main__":
    main()
