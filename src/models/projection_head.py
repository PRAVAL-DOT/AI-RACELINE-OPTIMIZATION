import torch
import torch.nn as nn

class ProjectionHead(nn.Module):
    """
    Non-linear Projection mapping downstream representations into a stable Hypersphere space.
    """
    def __init__(self, in_dim: int, out_dim: int = 128):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(in_dim, in_dim, bias=False),
            nn.BatchNorm1d(in_dim),
            nn.SiLU(),
            nn.Linear(in_dim, out_dim, bias=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.gate(x)
        return torch.nn.functional.normalize(x, p=2, dim=1)