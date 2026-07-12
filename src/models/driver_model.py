import torch
import torch.nn as nn
from typing import Dict, Tuple
from models.temporal_cnn import TemporalCNN
from models.transformer_encoder import TransformerEncoderModule
from models.projection_head import ProjectionHead

class F1StyleRepresentationModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.cnn = TemporalCNN(
            in_channels=config.FEATURE_DIM, 
            out_channels=config.CNN_FILTERS, 
            dropout=config.DROPOUT
        )
        
        self.transformer = TransformerEncoderModule(
            d_model=config.CNN_FILTERS,
            nhead=config.TRANSFORMER_NHEAD,
            num_layers=config.TRANSFORMER_LAYERS,
            dim_feedforward=config.DIM_FEEDFORWARD,
            dropout=config.DROPOUT
        )
        
        self.projection_head = ProjectionHead(
            in_dim=config.CNN_FILTERS, 
            out_dim=config.EMBEDDING_DIM
        )
        
        # Auxiliary Classification monitoring framework
        self.classifier_head = nn.Sequential(
            nn.Linear(config.CNN_FILTERS, config.CNN_FILTERS // 2),
            nn.LayerNorm(config.CNN_FILTERS // 2),
            nn.SiLU(),
            nn.Linear(config.CNN_FILTERS // 2, config.NUM_CLASSES)
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            Tuple[Embeddings, Logits] where:
                - Embeddings: L2 Normalized metrics space of shape (B, 128)
                - Logits: Unnormalized categorical output variables of shape (B, Num Drivers)
        """
        features = self.cnn(x)
        latent_vector = self.transformer(features)
        
        embeddings = self.projection_head(latent_vector)
        logits = self.classifier_head(latent_vector)
        
        return embeddings, logits