import os
from dataclasses import dataclass

@dataclass
class Config:
    # Path settings
    DATA_DIR: str = "data"
    MASTER_DATASET_PATH: str = os.path.join(DATA_DIR, "master_dataset_2025_laps.npy")
    LABELS_PATH: str = os.path.join(DATA_DIR, "driver_labels.npy")
    CHECKPOINT_DIR: str = "checkpoints"
    LOG_DIR: str = "logs"

    # Data Dimensions
    SEQ_LEN: int = 3000
    FEATURE_DIM: int = 11
    NUM_CLASSES: int = 20  # Overwritten dynamically during runtime loading by train.py

    # Model Hyperparameters
    CNN_FILTERS: int = 64
    TRANSFORMER_D_MODEL: int = 128
    TRANSFORMER_NHEAD: int = 4
    TRANSFORMER_LAYERS: int = 3
    DIM_FEEDFORWARD: int = 512
    EMBEDDING_DIM: int = 128
    DROPOUT: float = 0.1

    # Training Hyperparameters
    BATCH_SIZE: int = 32
    EPOCHS: int = 500          # Set to 500 maximum epochs as requested
    LR: float = 3e-4           # Finer learning rate for representation fine-tuning
    WEIGHT_DECAY: float = 1e-4
    TEMPERATURE: float = 0.05  # Lower temperature forces tighter style clusters
    
    # REPRESENTATION TUNING: Blends 30% Contrastive Clustering with 70% Cross-Entropy classification
    ALPHA: float = 0.3         
    
    GRAD_CLIP: float = 1.0
    EARLY_STOPPING_PATIENCE: int = 25  # Increased patience to let the 500-epoch run breathe