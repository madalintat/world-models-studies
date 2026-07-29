"""Linear controller: [z, h] -> 3 CarRacing actions.

Steering gets tanh (range -1..1), gas and brake get sigmoid (range 0..1).
With z_dim 32 and hidden_dim 256 this is (32 + 256 + 1) * 3 = 867 parameters,
small enough for CMA-ES to optimize directly without gradients.
"""

import numpy as np
import torch
import torch.nn as nn


class Controller(nn.Module):
    def __init__(self, z_dim: int = 32, hidden_dim: int = 256):
        super().__init__()
        self.z_dim = z_dim
        self.hidden_dim = hidden_dim
        self.fc = nn.Linear(z_dim + hidden_dim, 3)

    def forward(self, z: torch.Tensor, h: torch.Tensor):
        x = self.fc(torch.cat([z, h], dim=-1))
        steer = torch.tanh(x[..., 0])
        gas = torch.sigmoid(x[..., 1])
        brake = torch.sigmoid(x[..., 2])
        return torch.stack([steer, gas, brake], dim=-1)

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def get_flat_params(self) -> np.ndarray:
        with torch.no_grad():
            return torch.cat([p.view(-1) for p in self.parameters()]).numpy().copy()

    def set_flat_params(self, flat: np.ndarray):
        flat = torch.as_tensor(flat, dtype=torch.float32)
        i = 0
        with torch.no_grad():
            for p in self.parameters():
                n = p.numel()
                p.copy_(flat[i:i + n].view_as(p))
                i += n
        assert i == len(flat)


if __name__ == "__main__":
    c = Controller()
    print(f"controller parameters: {c.param_count()}")
