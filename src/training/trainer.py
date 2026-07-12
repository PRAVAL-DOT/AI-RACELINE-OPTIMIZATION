import os
import logging
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from typing import Tuple

# Root Config Import
from config import Config
from training.losses import SupervisedContrastiveLoss

logger = logging.getLogger("Trainer")

class EngineTrainer:
    def __init__(self, model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, config):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config

        self.contrastive_criterion = SupervisedContrastiveLoss(temperature=config.TEMPERATURE)
        self.classification_criterion = nn.CrossEntropyLoss()
        
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=config.LR, weight_decay=config.WEIGHT_DECAY)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=config.EPOCHS)
        self.scaler = GradScaler()
        
        self.writer = SummaryWriter(log_dir=config.LOG_DIR)
        os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
        
        self.best_loss = float('inf')
        self.patience_counter = 0

    def fit(self):
        logger.info(f"Initiating Training Engine deployment on: {self.device}")
        for epoch in range(1, self.config.EPOCHS + 1):
            train_loss, train_acc = self._train_epoch(epoch)
            val_loss, val_acc = self._val_epoch(epoch)
            
            self.scheduler.step()
            
            self.writer.add_scalar("Loss/Train", train_loss, epoch)
            self.writer.add_scalar("Accuracy/Train", train_acc, epoch)
            self.writer.add_scalar("Loss/Validation", val_loss, epoch)
            self.writer.add_scalar("Accuracy/Validation", val_acc, epoch)
            
            logger.info(f"Epoch {epoch:02d} | Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}%")
            
            if val_loss < self.best_loss:
                self.best_loss = val_loss
                self.patience_counter = 0
                torch.save(self.model.state_dict(), os.path.join(self.config.CHECKPOINT_DIR, "best_driver_style_model.pt"))
                logger.info("  ↳ Saved updated champion checkpoint.")
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.config.EARLY_STOPPING_PATIENCE:
                    logger.warning(f"Early stop mechanism reached validation freeze threshold. Exiting.")
                    break
        self.writer.close()

    def _train_epoch(self, epoch: int) -> Tuple[float, float]:
        self.model.train()
        total_loss, correct_preds, total_items = 0.0, 0, 0
        
        for x, y in self.train_loader:
            x, y = x.to(self.device), y.to(self.device)
            self.optimizer.zero_grad()
            
            with autocast(device_type=self.device.type):
                embeddings, logits = self.model(x)
                loss_supcon = self.contrastive_criterion(embeddings, y)
                loss_ce = self.classification_criterion(logits, y)
                loss = (self.config.ALPHA * loss_supcon) + ((1.0 - self.config.ALPHA) * loss_ce)
                
            self.scaler.scale(loss).backward()
            if self.config.GRAD_CLIP > 0:
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.config.GRAD_CLIP)
                
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            total_loss += loss.item() * x.size(0)
            _, predicted = torch.max(logits, 1)
            correct_preds += (predicted == y).sum().item()
            total_items += y.size(0)
            
        return total_loss / total_items, (correct_preds / total_items) * 100

    @torch.no_grad()
    def _val_epoch(self, epoch: int) -> Tuple[float, float]:
        self.model.eval()
        total_loss, correct_preds, total_items = 0.0, 0, 0
        
        for x, y in self.val_loader:
            x, y = x.to(self.device), y.to(self.device)
            
            with autocast(device_type=self.device.type):
                embeddings, logits = self.model(x)
                loss_supcon = self.contrastive_criterion(embeddings, y)
                loss_ce = self.classification_criterion(logits, y)
                loss = (self.config.ALPHA * loss_supcon) + ((1.0 - self.config.ALPHA) * loss_ce)
                
            total_loss += loss.item() * x.size(0)
            _, predicted = torch.max(logits, 1)
            correct_preds += (predicted == y).sum().item()
            total_items += y.size(0)
            
        return total_loss / total_items, (correct_preds / total_items) * 100