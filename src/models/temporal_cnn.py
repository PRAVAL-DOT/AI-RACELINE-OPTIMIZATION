import torch
import torch.nn as nn

class TemporalCNN(nn.Module):
    """
    1D Temporal Convolutional Neural Network designed to extract localized driver
    style semantics (braking modulation, micro throttle inputs, jerk behaviors).
    """
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.1):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.SiLU(),
            nn.Dropout(dropout),
            
            nn.Conv1d(out_channels, out_channels, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.SiLU(),
            nn.Dropout(dropout),
            
            nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.SiLU()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Expected input shape: (Batch, Seq_Len, Feature_Dim) -> Permute for Conv1D: (Batch, Feature_Dim, Seq_Len)
        x = x.permute(0, 2, 1)
        x = self.network(x)
        # Permute back to: (Batch, New_Seq_Len, Out_Channels)
        return x.permute(0, 2, 1)