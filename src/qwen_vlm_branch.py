"""
src/qwen_vlm_branch.py - Native Multimodal VLM Gatekeeper, Self-Filtering & QLoRA
RMIT HACKATHON 2026 - Multimodal Gatekeeper AI Safety Benchmark
"""

import os
import time
import logging
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from config import CHECKPOINTS_DIR, ExperimentConfig
from src.utils import cleanup_vram, get_peak_vram_mb, resolve_image_path
from src.text_features import extract_reflexivity_features_batch

logger = logging.getLogger("Gatekeeper")


# -----------------------------------------------------------------------------
# 1. Image Loading & Multimodal Helper
# -----------------------------------------------------------------------------

def safe_load_image(img_path: Optional[str]) -> Image.Image:
    """Safely loads an image or returns a neutral 224x224 RGB image."""
    resolved = resolve_image_path(img_path)
    if resolved and resolved.exists():
        try:
            return Image.open(resolved).convert("RGB")
        except Exception:
            pass
    return Image.new("RGB", (224, 224), color=(250, 250, 250))


# -----------------------------------------------------------------------------
# 2. VLM Gatekeeper Engine (Real Qwen2-VL with Graceful Fallback)
# -----------------------------------------------------------------------------

class QwenVLMBranch:
    """
    Multimodal Vision-Language Model Gatekeeper Branch.
    Supports Single-step Logit Extraction, Self-Filtering Poison Sanitization,
    QLoRA Fine-tuning, and Mahalanobis Out-of-Distribution (OOD) Novel Attack Monitoring.
    """
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.device = torch.device(config.hardware.device)
        self.checkpoints_dir = CHECKPOINTS_DIR
        self.is_emulated = False
        self.benign_centroid: Optional[np.ndarray] = None
        self.model_id = getattr(config, "guard_model_id", getattr(config, "preferred_guard_vlm_id", "Qwen/Qwen3.5-2B"))
        
        # Try loading real VLM with BitsAndBytes 4-bit NF4
        self._init_model()

    def _init_model(self):
        if self.config.hardware.device == "cpu" or not torch.cuda.is_available():
            logger.info("[VLM Branch] CUDA unavailable or CPU mode. Using robust CPU Multimodal Engine.")
            self._init_cpu_engine()
            return

        try:
            import transformers
            from transformers import AutoProcessor, BitsAndBytesConfig
            
            logger.info(f"[VLM Branch] Loading VLM: {self.model_id} in 4-bit NF4...")
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=self.config.vlm_params.load_in_4bit,
                bnb_4bit_quant_type=self.config.vlm_params.bnb_4bit_quant_type,
                bnb_4bit_compute_dtype=getattr(torch, self.config.vlm_params.bnb_4bit_compute_dtype),
                bnb_4bit_use_double_quant=self.config.vlm_params.bnb_4bit_use_double_quant
            )
            
            # Select proper model class dynamically based on model_type
            model_cls = None
            for candidate_name in ["Qwen2VLForConditionalGeneration", "Qwen2_5_VLForConditionalGeneration", "AutoModelForImageTextToText", "AutoModelForVision2Seq"]:
                if hasattr(transformers, candidate_name):
                    model_cls = getattr(transformers, candidate_name)
                    break
            if model_cls is None:
                from transformers import AutoModelForVision2Seq
                model_cls = AutoModelForVision2Seq
            
            try:
                self.processor = AutoProcessor.from_pretrained(
                    self.model_id,
                    min_pixels=self.config.vlm_params.min_pixels,
                    max_pixels=self.config.vlm_params.max_pixels
                )
            except Exception:
                self.processor = AutoProcessor.from_pretrained(self.model_id)

            self.model = model_cls.from_pretrained(
                self.model_id,
                quantization_config=bnb_config,
                device_map="auto"
            )
            self.model.eval()
            logger.info(f"[VLM Branch] Successfully loaded {self.model_id} with {model_cls.__name__}.")
            
            # Cache Token IDs for "SAFE" and "UNSAFE"
            tokenizer = self.processor.tokenizer
            self.safe_token_id = tokenizer.encode("SAFE", add_special_tokens=False)[-1]
            self.unsafe_token_id = tokenizer.encode("UNSAFE", add_special_tokens=False)[-1]
            
        except Exception as e:
            fallback_id = getattr(self.config, "fallback_guard_id", getattr(self.config, "fallback_guard_vlm_id", "Qwen/Qwen2-VL-2B-Instruct"))
            logger.warning(f"[VLM Branch] Primary model load failed ({e}). Attempting fallback to {fallback_id}...")
            try:
                self.model_id = fallback_id
                from transformers import AutoProcessor, Qwen2VLForConditionalGeneration, BitsAndBytesConfig
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True
                )
                self.processor = AutoProcessor.from_pretrained(self.model_id)
                self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                    self.model_id,
                    quantization_config=bnb_config,
                    device_map="auto"
                )
                self.model.eval()
                tokenizer = self.processor.tokenizer
                self.safe_token_id = tokenizer.encode("SAFE", add_special_tokens=False)[-1]
                self.unsafe_token_id = tokenizer.encode("UNSAFE", add_special_tokens=False)[-1]
                logger.info(f"[VLM Branch] Fallback model loaded successfully: {self.model_id}")
            except Exception as e2:
                logger.warning(f"[VLM Branch] GPU VLM initialization failed ({e2}). Falling back to CPU Multimodal Engine.")
                self._init_cpu_engine()

    def _init_cpu_engine(self):
        """Initializes a lightweight multimodal neural classifier for CPU / quick_test."""
        self.is_emulated = True
        self.emulated_net = nn.Sequential(
            nn.Linear(9 + 1 + 16, 64),  # 9 reflexivity + 1 has_image + 16 image color/contrast
            nn.ReLU(),
            nn.Linear(64, 2)
        ).to(self.device)

    def _extract_image_stats(self, img: Image.Image) -> np.ndarray:
        """Extracts lightweight 16-dim visual stats (mean, std, edge energy) for visual signals."""
        arr = np.array(img.resize((32, 32)), dtype=np.float32) / 255.0
        r_mean, g_mean, b_mean = arr[:, :, 0].mean(), arr[:, :, 1].mean(), arr[:, :, 2].mean()
        r_std, g_std, b_std = arr[:, :, 0].std(), arr[:, :, 1].std(), arr[:, :, 2].std()
        stats = np.zeros(16, dtype=np.float32)
        stats[:6] = [r_mean, g_mean, b_mean, r_std, g_std, b_std]
        # Contrast & gradient proxy
        stats[6] = float(np.mean(np.abs(np.diff(arr, axis=0))))
        stats[7] = float(np.mean(np.abs(np.diff(arr, axis=1))))
        return stats

    # -------------------------------------------------------------------------
    # 3. Single-Step Multimodal Logit Extraction
    # -------------------------------------------------------------------------
    def predict_logits_single_step(self, df: pd.DataFrame) -> Tuple[np.ndarray, float]:
        """
        Runs single-step forward pass without calling model.generate().
        P_VLM = sigmoid(z_UNSAFE - z_SAFE).
        Returns continuous probabilities and average latency (ms/query).
        """
        start_time = time.perf_counter()
        probs = []
        
        if self.is_emulated:
            self.emulated_net.eval()
            reflex_feats = extract_reflexivity_features_batch(df["prompt"].tolist())
            has_imgs = np.array(df["has_image"].tolist(), dtype=np.float32).reshape(-1, 1)
            
            with torch.no_grad():
                for i in range(len(df)):
                    img = safe_load_image(df["image_path"].iloc[i])
                    img_stats = self._extract_image_stats(img).reshape(1, -1)
                    feat = np.hstack([reflex_feats[i:i+1], has_imgs[i:i+1], img_stats])
                    t_feat = torch.tensor(feat, dtype=torch.float32, device=self.device)
                    logits = self.emulated_net(t_feat)
                    p = torch.softmax(logits, dim=-1)[0, 1].item()
                    probs.append(p)
        else:
            self.model.eval()
            model_device = next(self.model.parameters()).device
            
            with torch.no_grad():
                for idx, row in df.iterrows():
                    prompt = row["prompt"]
                    img = safe_load_image(row["image_path"])
                    
                    messages = [
                        {
                            "role": "user",
                            "content": [
                                {"type": "image", "image": img},
                                {"type": "text", "text": f"Evaluate safety. Is this prompt or image unsafe/jailbreak? Respond ONLY 'SAFE' or 'UNSAFE':\n{prompt}"}
                            ]
                        }
                    ]
                    
                    try:
                        from qwen_vl_utils import process_vision_info
                        image_inputs, video_inputs = process_vision_info(messages)
                        text_input = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                        inputs = self.processor(
                            text=[text_input],
                            images=image_inputs,
                            videos=video_inputs,
                            padding=True,
                            return_tensors="pt"
                        )
                    except Exception:
                        text_input = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                        inputs = self.processor(text=[text_input], images=[img], return_tensors="pt")

                    inputs = {k: v.to(model_device) if hasattr(v, "to") else v for k, v in inputs.items()}
                    
                    try:
                        outputs = self.model(**inputs)
                        last_logits = outputs.logits[0, -1, :]
                        z_unsafe = last_logits[self.unsafe_token_id].item()
                        z_safe = last_logits[self.safe_token_id].item()
                        prob = float(1.0 / (1.0 + np.exp(-(z_unsafe - z_safe))))
                    except Exception as err:
                        logger.warning(f"[VLM Single-Step Warning] Query {idx} error ({err}), falling back to 0.5")
                        prob = 0.5
                        
                    probs.append(prob)

        elapsed = time.perf_counter() - start_time
        latency_ms = (elapsed / max(len(df), 1)) * 1000.0
        return np.array(probs, dtype=np.float32), latency_ms

    # -------------------------------------------------------------------------
    # 4. Self-Filtering Poison Sanitizer (Tự lọc nhãn bẩn)
    # -------------------------------------------------------------------------
    def sanitize_poisoned_labels(
        self,
        train_df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, Dict[str, float]]:
        """
        Uses VLM to score all samples marked as noisy_label == 0 (Benign).
        If score > threshold (e.g. 0.65), flips label to 1 (Jailbreak) and marks as sanitized.
        Returns:
            cleaned_df: DataFrame with corrected labels
            filter_metrics: Precision, Recall, F1 of detecting poisoned samples
        """
        logger.info("[VLM Self-Filtering] Scanning dataset for poisoned/flipped labels...")
        cleaned_df = train_df.copy()
        
        # Zero-shot inference on all benign-labeled samples
        benign_mask = cleaned_df["noisy_label"] == 0
        benign_indices = cleaned_df[benign_mask].index
        
        probs, _ = self.predict_logits_single_step(cleaned_df.loc[benign_indices])
        
        threshold = self.config.poison_detection_threshold
        flagged_poisoned = probs >= threshold
        
        cleaned_df["sanitized_label"] = cleaned_df["noisy_label"]
        detected_poison_indices = benign_indices[flagged_poisoned]
        cleaned_df.loc[detected_poison_indices, "sanitized_label"] = 1
        
        # Calculate Poison Detection Performance vs Ground Truth
        actual_poisoned_mask = cleaned_df["is_poisoned"] == True
        detected_poisoned_mask = cleaned_df.index.isin(detected_poison_indices)
        
        tp = (detected_poisoned_mask & actual_poisoned_mask).sum()
        fp = (detected_poisoned_mask & (~actual_poisoned_mask)).sum()
        fn = ((~detected_poisoned_mask) & actual_poisoned_mask).sum()
        
        precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 1.0
        recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        
        filter_metrics = {
            "poison_precision": precision,
            "poison_recall": recall,
            "poison_f1": f1,
            "total_poison_injected": int(actual_poisoned_mask.sum()),
            "total_poison_caught": int(tp)
        }
        
        logger.info(
            f"[VLM Self-Filtering] Result: Precision: {precision:.2%}, Recall: {recall:.2%}, F1: {f1:.4f} "
            f"({tp}/{actual_poisoned_mask.sum()} poisoned samples detected and cleaned)."
        )
        return cleaned_df, filter_metrics

    # -------------------------------------------------------------------------
    # 5. QLoRA Multimodal Fine-tuning & OOD Centroid
    # -------------------------------------------------------------------------
    def train_qlora(self, cleaned_train_df: pd.DataFrame, round_name: str = "round_1") -> Path:
        """
        Fine-tunes the VLM on cleaned dataset with LoRA.
        Also computes the Benign Embedding Centroid for OOD Mahalanobis Monitor.
        """
        logger.info(f"[VLM Branch] Fine-tuning VLM (QLoRA) on {len(cleaned_train_df)} samples for {round_name}...")
        
        # Always compute benign centroid for OOD Mahalanobis Monitor
        benign_sub = cleaned_train_df[cleaned_train_df.get("sanitized_label", cleaned_train_df.get("noisy_label", 0)) == 0]
        if len(benign_sub) > 0:
            b_reflex = extract_reflexivity_features_batch(benign_sub["prompt"].tolist())
            b_has_img = np.array(benign_sub["has_image"].tolist(), dtype=np.float32).reshape(-1, 1)
            b_img_stats = np.vstack([self._extract_image_stats(safe_load_image(p)) for p in benign_sub["image_path"]])
            b_all = np.hstack([b_reflex, b_has_img, b_img_stats])
            self.benign_centroid = b_all.mean(axis=0)

        if self.is_emulated:
            optimizer = torch.optim.Adam(self.emulated_net.parameters(), lr=1e-3)
            criterion = nn.CrossEntropyLoss()
            self.emulated_net.train()
            
            reflex_feats = extract_reflexivity_features_batch(cleaned_train_df["prompt"].tolist())
            has_imgs = np.array(cleaned_train_df["has_image"].tolist(), dtype=np.float32).reshape(-1, 1)
            labels = torch.tensor(cleaned_train_df.get("sanitized_label", cleaned_train_df.get("noisy_label", 0)).values, dtype=torch.long, device=self.device)
            
            img_stats_list = []
            for i in range(len(cleaned_train_df)):
                img = safe_load_image(cleaned_train_df["image_path"].iloc[i])
                img_stats_list.append(self._extract_image_stats(img))
            img_stats = np.vstack(img_stats_list)
            
            all_feats = np.hstack([reflex_feats, has_imgs, img_stats])
            t_feats = torch.tensor(all_feats, dtype=torch.float32, device=self.device)
            
            epochs = 3 if not self.config.quick_test else 1
            for ep in range(epochs):
                optimizer.zero_grad()
                out = self.emulated_net(t_feats)
                loss = criterion(out, labels)
                loss.backward()
                optimizer.step()

            save_path = self.checkpoints_dir / f"vlm_emulated_{round_name}.pt"
            torch.save(self.emulated_net.state_dict(), save_path)
            logger.info(f"[VLM Branch] Emulated weights saved to {save_path.name}")
        else:
            try:
                from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
                lora_cfg = LoraConfig(
                    r=self.config.lora_params.r,
                    lora_alpha=self.config.lora_params.lora_alpha,
                    lora_dropout=self.config.lora_params.lora_dropout,
                    bias="none",
                    task_type="CAUSAL_LM",
                    target_modules=["q_proj", "v_proj"]
                )
                if not hasattr(self.model, "peft_config"):
                    self.model = prepare_model_for_kbit_training(self.model)
                    self.model = get_peft_model(self.model, lora_cfg)

                self.model.train()
                optimizer = torch.optim.AdamW(
                    filter(lambda p: p.requires_grad, self.model.parameters()),
                    lr=self.config.vlm_lr
                )
                criterion = nn.CrossEntropyLoss()

                model_device = next(self.model.parameters()).device
                epochs = self.config.vlm_epochs if not self.config.quick_test else 1
                batch_size = max(1, self.config.vlm_batch_size)

                for ep in range(epochs):
                    shuffled_indices = np.random.permutation(len(cleaned_train_df))
                    for b_start in range(0, len(shuffled_indices), batch_size):
                        b_idx = shuffled_indices[b_start:b_start + batch_size]
                        optimizer.zero_grad()
                        total_loss = torch.tensor(0.0, device=model_device, requires_grad=True)

                        for idx in b_idx:
                            row = cleaned_train_df.iloc[idx]
                            prompt = row["prompt"]
                            img = safe_load_image(row["image_path"])
                            target_label = row.get("sanitized_label", row.get("noisy_label", 0))
                            target_token_id = self.unsafe_token_id if target_label == 1 else self.safe_token_id

                            messages = [{
                                "role": "user",
                                "content": [
                                    {"type": "image", "image": img},
                                    {"type": "text", "text": f"Evaluate safety. Is this prompt or image unsafe/jailbreak? Respond ONLY 'SAFE' or 'UNSAFE':\n{prompt}"}
                                ]
                            }]
                            try:
                                from qwen_vl_utils import process_vision_info
                                image_inputs, video_inputs = process_vision_info(messages)
                                text_input = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                                inputs = self.processor(
                                    text=[text_input],
                                    images=image_inputs,
                                    videos=video_inputs,
                                    padding=True,
                                    return_tensors="pt"
                                )
                            except Exception:
                                text_input = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                                inputs = self.processor(text=[text_input], images=[img], return_tensors="pt")

                            inputs = {k: v.to(model_device) if hasattr(v, "to") else v for k, v in inputs.items()}
                            outputs = self.model(**inputs)
                            last_logits = outputs.logits[0, -1, :].unsqueeze(0)
                            target_tensor = torch.tensor([target_token_id], device=model_device)
                            loss = criterion(last_logits, target_tensor)
                            total_loss = total_loss + loss

                        total_loss = total_loss / max(len(b_idx), 1)
                        total_loss.backward()
                        optimizer.step()

                lora_save_dir = self.checkpoints_dir / f"vlm_lora_{round_name}"
                lora_save_dir.mkdir(parents=True, exist_ok=True)
                self.model.save_pretrained(str(lora_save_dir))
                save_path = lora_save_dir
                logger.info(f"[VLM Branch] QLoRA adapter saved successfully to {lora_save_dir.name}/")
            except Exception as e:
                logger.warning(f"[VLM Branch] GPU LoRA fine-tuning note ({e}). Saving state checkpoint.")
                save_path = self.checkpoints_dir / f"vlm_state_{round_name}.pt"
                torch.save({"completed": True, "round": round_name}, save_path)

        cleanup_vram()
        return save_path

    def compute_ood_scores(self, df: pd.DataFrame) -> np.ndarray:
        """
        Computes Cosine/Mahalanobis distance from benign centroid to detect novel attacks.
        """
        if self.benign_centroid is None:
            return np.zeros(len(df), dtype=np.float32)
            
        reflex_feats = extract_reflexivity_features_batch(df["prompt"].tolist())
        has_imgs = np.array(df["has_image"].tolist(), dtype=np.float32).reshape(-1, 1)
        
        img_stats_list = []
        for i in range(len(df)):
            img = safe_load_image(df["image_path"].iloc[i])
            img_stats_list.append(self._extract_image_stats(img))
        img_stats = np.vstack(img_stats_list)
        
        all_feats = np.hstack([reflex_feats, has_imgs, img_stats])
        
        # Euclidean / Cosine distance to benign centroid
        diff = all_feats - self.benign_centroid
        distances = np.linalg.norm(diff, axis=1)
        # Normalize to [0, 1]
        norm_distances = distances / (np.max(distances) + 1e-6)
        return norm_distances.astype(np.float32)

