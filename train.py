import os
import sys
import logging
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split

# Inject 'src' into system paths to cleanly manage localized sub-package loads
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Root layout loads
from config import Config

# Isolated submodule package assets
from models.driver_model import F1StyleRepresentationModel
from training.dataset import F1TelemetryDataset
from training.trainer import EngineTrainer
from utils.metrics import evaluate_embeddings_topk
from utils.visualization import generate_style_visualizations

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("PipelineExecutive")

def run_pipeline():
    cfg = Config()
    
    if not os.path.exists(cfg.MASTER_DATASET_PATH) or not os.path.exists(cfg.LABELS_PATH):
        raise FileNotFoundError(
            f"Missing base source dataset matrices. Run the extraction pipeline first to generate: "
            f"{cfg.MASTER_DATASET_PATH} and {cfg.LABELS_PATH}"
        )

    logger.info("Instantiating Telemetry Data Loaders...")
    full_dataset = F1TelemetryDataset(cfg.MASTER_DATASET_PATH, cfg.LABELS_PATH)
    
    # Calculate unique drivers present to prevent cross-entropy target indexing bugs
    unique_drivers = np.unique(full_dataset.labels)
    cfg.NUM_CLASSES = int(len(unique_drivers))
    logger.info(f"Detected {cfg.NUM_CLASSES} unique drivers in dataset (Max Label ID: {np.max(full_dataset.labels)})")
    
    if np.max(full_dataset.labels) >= cfg.NUM_CLASSES:
        raise ValueError(
            f"Driver labels are non-contiguous. Labels must scale cleanly from 0 to {cfg.NUM_CLASSES - 1}. "
            f"Please verify categorical serialization in your upstream extraction pipeline."
        )
    
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size], 
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_dataset, batch_size=cfg.BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg.BATCH_SIZE, shuffle=False)

    logger.info("Initializing Hybrid Temporal CNN-Transformer Architecture...")
    model = F1StyleRepresentationModel(config=cfg)

    trainer = EngineTrainer(model=model, train_loader=train_loader, val_loader=val_loader, config=cfg)
    trainer.fit()

    logger.info("Extracting final driver style fingerprints using champion model weights...")
    best_model_path = os.path.join(cfg.CHECKPOINT_DIR, "best_driver_style_model.pt")
    model.load_state_dict(torch.load(best_model_path))
    model.eval()
    
    all_embeddings = []
    all_labels = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    with torch.no_grad():
        for idx in range(len(full_dataset)):
            x, y = full_dataset[idx]
            x = x.unsqueeze(0).to(device)
            emb, _ = model(x)
            all_embeddings.append(emb.cpu().numpy().flatten())
            all_labels.append(y.item())
            
    embeddings_arr = np.array(all_embeddings)
    labels_arr = np.array(all_labels)

    fingerprint_output_path = os.path.join(cfg.DATA_DIR, "driver_fingerprint_embeddings.npy")
    np.save(fingerprint_output_path, embeddings_arr)
    logger.info(f"Saved {embeddings_arr.shape[0]} unique driver style embeddings to: {fingerprint_output_path}")

    metrics = evaluate_embeddings_topk(embeddings_arr, labels_arr, topk=(1, 5))
    print("\n" + "="*50 + "\n FINAL LATENT RECOGNITION SPACE EVALUATION \n" + "="*50)
    for k, v in metrics.items():
        print(f" * {k}: {v:.2f}%")
    print("="*50)

    generate_style_visualizations(
        embeddings=embeddings_arr, 
        labels=labels_arr, 
        mapping_csv_path=os.path.join(cfg.DATA_DIR, "label_mapping.csv")
    )

if __name__ == "__main__":
    run_pipeline()