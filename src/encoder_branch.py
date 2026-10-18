"""
src/encoder_branch.py - Fast Multilingual Encoder with Explicit Reflexivity Features
RMIT HACKATHON 2026 - Multimodal Gatekeeper AI Safety Benchmark
"""

import time
import logging
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

try:
    from transformers import AutoTokenizer, AutoModel, AutoConfig
    HAS_TRANSFORMERS = True
except ImportError:
    AutoTokenizer, AutoModel, AutoConfig = None, None, None
    HAS_TRANSFORMERS = False

from config import CHECKPOINTS_DIR, ExperimentConfig
from src.utils import cleanup_vram, get_peak_vram_mb
from src.text_features import extract_reflexivity_features_batch, get_reflexivity_feature_dim

logger = logging.getLogger("Gatekeeper")


# -----------------------------------------------------------------------------
# 1. Dataset & Collate Fn
# -----------------------------------------------------------------------------

class FastEncoderDataset(Dataset):
    def __init__(self, texts: List[str], has_images: List[bool], labels: Optional[List[int]] = None):
        self.texts = texts
        self.has_images = has_images
        self.labels = labels
        # Precompute reflexivity features
        self.reflex_features = extract_reflexivity_features_batch(texts)
        # Append has_image as 10th feature
        has_img_col = np.array(has_images, dtype=np.float32).reshape(-1, 1)
        self.meta_features = np.hstack([self.reflex_features, has_img_col])

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = {
            "text": self.texts[idx],
            "meta_feat": torch.tensor(self.meta_features[idx], dtype=torch.float32)
        }
        if self.labels is not None:
            item["label"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


# -----------------------------------------------------------------------------
# 2. Hybrid Encoder Architecture (Transformer + Reflexivity MLP)
# -----------------------------------------------------------------------------

class HybridEncoderClassifier(nn.Module):
    def __init__(self, model_name: str, meta_dim: int = 10, num_labels: int = 2):
        super().__init__()
        self.model_name = model_name
        self.config = AutoConfig.from_pretrained(model_name)
        self.transformer = AutoModel.from_pretrained(model_name, config=self.config)
        hidden_size = self.config.hidden_size

        # Meta-features projection layer
        self.meta_proj = nn.Sequential(
            nn.Linear(meta_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        # Combined Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size + 64, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_labels)
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        meta_feat: torch.Tensor
    ) -> torch.Tensor:
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        # Use mean pooling or [CLS] token
        cls_rep = outputs.last_hidden_state[:, 0, :]
        meta_rep = self.meta_proj(meta_feat)
        combined = torch.cat([cls_rep, meta_rep], dim=-1)
        logits = self.classifier(combined)
        return logits


class LightweightFallbackClassifier(nn.Module):
    """
    Lightweight fallback model for ultra-fast local testing or offline environments.
    """
    def __init__(self, meta_dim: int = 10, num_labels: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(meta_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_labels)
        )

    def forward(
        self,
        input_ids: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        meta_feat: torch.Tensor
    ) -> torch.Tensor:
        return self.net(meta_feat)


# -----------------------------------------------------------------------------
# 3. Training & Inference Coordinator
# -----------------------------------------------------------------------------

class EncoderBranchTrainer:
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.device = torch.device(config.hardware.device)
        self.checkpoints_dir = CHECKPOINTS_DIR
        self.meta_dim = get_reflexivity_feature_dim() + 1  # 9 + 1 (has_image) = 10
        
        # Load Tokenizer & Model
        self.model_name = config.preferred_encoder_id
        self.is_fallback = False
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = HybridEncoderClassifier(self.model_name, meta_dim=self.meta_dim)
            logger.info(f"[Encoder Branch] Loaded Transformer backbone: {self.model_name}")
        except Exception as e:
            logger.warning(f"[Encoder Branch] Could not load {self.model_name} ({e}). Using LightweightFallbackClassifier.")
            self.is_fallback = True
            self.tokenizer = None
            self.model = LightweightFallbackClassifier(meta_dim=self.meta_dim)

        self.model.to(self.device)

    def train(self, train_df: pd.DataFrame, round_name: str = "round_1") -> Path:
        """
        Trains the Fast Encoder on the provided DataFrame and saves checkpoint.
        """
        logger.info(f"[Encoder Branch] Training Fast Encoder for {round_name} on {len(train_df)} samples...")
        self.model.train()
        
        texts = train_df["prompt"].tolist()
        has_images = train_df["has_image"].tolist()
        labels = train_df["noisy_label"].tolist()
        
        dataset = FastEncoderDataset(texts, has_images, labels)
        batch_size = self.config.encoder_batch_size
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.config.encoder_lr, weight_decay=0.01)
        criterion = nn.CrossEntropyLoss()
        
        epochs = self.config.encoder_epochs if not self.config.quick_test else 1
        
        for epoch in range(1, epochs + 1):
            total_loss = 0.0
            for batch in dataloader:
                optimizer.zero_grad()
                meta_feat = batch["meta_feat"].to(self.device)
                batch_labels = batch["label"].to(self.device)
                
                if not self.is_fallback and self.tokenizer is not None:
                    encoded = self.tokenizer(
                        batch["text"],
                        padding=True,
                        truncation=True,
                        max_length=self.config.encoder_max_length,
                        return_tensors="pt"
                    ).to(self.device)
                    logits = self.model(
                        input_ids=encoded["input_ids"],
                        attention_mask=encoded["attention_mask"],
                        meta_feat=meta_feat
                    )
                else:
                    logits = self.model(None, None, meta_feat)
                    
                loss = criterion(logits, batch_labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                
            avg_loss = total_loss / max(len(dataloader), 1)
            logger.info(f"[Encoder Branch] {round_name} Epoch {epoch}/{epochs} - Loss: {avg_loss:.4f}")

        # Save checkpoint
        save_path = self.checkpoints_dir / f"encoder_{round_name}.pt"
        torch.save(self.model.state_dict(), save_path)
        logger.info(f"[Encoder Branch] Saved checkpoint to {save_path.name}")
        cleanup_vram()
        return save_path

    def predict(self, df: pd.DataFrame) -> Tuple[np.ndarray, float]:
        """
        Inference on test DataFrame.
        Returns:
            probabilities: continuous P(Jailbreak) in [0, 1]
            latency_ms: average latency per query in milliseconds
        """
        self.model.eval()
        texts = df["prompt"].tolist()
        has_images = df["has_image"].tolist()
        
        dataset = FastEncoderDataset(texts, has_images)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
        
        all_probs = []
        start_time = time.perf_counter()
        
        with torch.no_grad():
            for batch in dataloader:
                meta_feat = batch["meta_feat"].to(self.device)
                
                if not self.is_fallback and self.tokenizer is not None:
                    encoded = self.tokenizer(
                        batch["text"],
                        padding=True,
                        truncation=True,
                        max_length=self.config.encoder_max_length,
                        return_tensors="pt"
                    ).to(self.device)
                    logits = self.model(
                        input_ids=encoded["input_ids"],
                        attention_mask=encoded["attention_mask"],
                        meta_feat=meta_feat
                    )
                else:
                    logits = self.model(None, None, meta_feat)
                    
                probs = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
                all_probs.extend(probs)
                
        elapsed_sec = time.perf_counter() - start_time
        latency_ms = (elapsed_sec / max(len(df), 1)) * 1000.0
        
        return np.array(all_probs, dtype=np.float32), latency_ms
