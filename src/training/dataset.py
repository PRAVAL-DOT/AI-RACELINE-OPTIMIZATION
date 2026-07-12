import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Tuple

class F1TelemetryDataset(Dataset):
    def __init__(self, features_path: str, labels_path: str):
        self.features = np.load(features_path).astype(np.float32)
        self.labels = np.load(labels_path).astype(np.int64)
        
        if len(self.features) != len(self.labels):
            raise ValueError(f"Feature count ({len(self.features)}) differs from label counts ({len(self.labels)}).")

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x_tensor = torch.from_numpy(self.features[idx])
        y_tensor = torch.tensor(self.labels[idx], dtype=torch.long)
        return x_tensor, y_tensor