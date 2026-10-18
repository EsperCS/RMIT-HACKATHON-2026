"""
config.py - Centralized Configuration & Environment Auto-Detection
RMIT HACKATHON 2026 - Multimodal Gatekeeper AI Safety Benchmark
"""

import os
import sys
import torch
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

# Enforce unbuffered output for real-time terminal logs on Colab/CLI
os.environ["PYTHONUNBUFFERED"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Base Directory Resolution
BASE_DIR = Path(__file__).resolve().parent

# Directory Paths
DATA_DIR = BASE_DIR / "data"
DOWNLOADS_DIR = DATA_DIR / "downloads"
SYNTHETIC_IMAGES_DIR = DATA_DIR / "synthetic_images"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
CHECKPOINTS_DIR = BASE_DIR / "checkpoints"
RESULTS_DIR = BASE_DIR / "results"

# Ensure all primary directories exist
for directory in [DATA_DIR, DOWNLOADS_DIR, SYNTHETIC_IMAGES_DIR, PROCESSED_DATA_DIR, CHECKPOINTS_DIR, RESULTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Colab Google Drive Auto-Detection Path
COLAB_DRIVE_PATH = Path("/content/drive/MyDrive")
DEFAULT_DRIVE_BACKUP_SUBDIR = "RMIT_Hackathon_2026_Backup"


def detect_colab_drive_backup_dir(custom_path: Optional[str] = None) -> Optional[Path]:
    """
    Auto-detects if Google Drive is mounted on Colab.
    Returns path to backup directory on Drive if available.
    """
    if custom_path:
        custom_p = Path(custom_path)
        try:
            custom_p.mkdir(parents=True, exist_ok=True)
            return custom_p
        except Exception as e:
            logging.warning(f"Failed to use custom drive path {custom_path}: {e}")

    if COLAB_DRIVE_PATH.exists() and COLAB_DRIVE_PATH.is_dir():
        backup_p = COLAB_DRIVE_PATH / DEFAULT_DRIVE_BACKUP_SUBDIR
        try:
            backup_p.mkdir(parents=True, exist_ok=True)
            return backup_p
        except Exception as e:
            logging.warning(f"Google Drive detected but failed to create backup dir: {e}")
    return None


@dataclass
class HardwareConfig:
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    gpu_name: str = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    vram_gb: float = (
        torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        if torch.cuda.is_available()
        else 0.0
    )
    is_colab_t4: bool = False
    is_colab_l4: bool = False

    def __post_init__(self):
        if "T4" in self.gpu_name:
            self.is_colab_t4 = True
        elif "L4" in self.gpu_name:
            self.is_colab_l4 = True


@dataclass
class VLMConfig:
    """
    VLM Token & Visual Resolution constraints to prevent OOM on 15GB T4 GPU.
    """
    # 28 * 28 = 784 pixels per visual patch token
    min_pixels: int = 128 * 28 * 28   # 100,352 (~128 visual tokens)
    max_pixels: int = 384 * 28 * 28   # 301,056 (~384 visual tokens)
    max_new_tokens: int = 16
    load_in_4bit: bool = True
    bnb_4bit_compute_dtype: str = "bfloat16" if torch.cuda.is_bf16_supported() else "float16"
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True


@dataclass
class LoRAConfig:
    r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    bias: str = "none"
    task_type: str = "CAUSAL_LM"
    target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"]
    )


@dataclass
class ExperimentConfig:
    # Reproducibility
    seed: int = 42

    # Model IDs with robust fallback
    preferred_encoder_id: str = "FacebookAI/xlm-roberta-base"
    fallback_encoder_id: str = "microsoft/deberta-v3-base"

    preferred_guard_vlm_id: str = "Qwen/Qwen2.5-VL-3B-Instruct"
    fallback_guard_vlm_id: str = "Qwen/Qwen2-VL-2B-Instruct"
    cpu_mock_guard_vlm_id: str = "Qwen/Qwen2-VL-2B-Instruct"

    preferred_target_model_id: str = "Qwen/Qwen2.5-3B-Instruct"
    fallback_target_model_id: str = "Qwen/Qwen2.5-1.5B-Instruct"

    # Execution Mode
    quick_test: bool = False
    full_run: bool = False
    resume: bool = True
    stage: str = "all"  # 'all', 'data', 'encoder', 'vlm', 'e2e', 'eval'

    # Dataset & Poisoning Settings
    val_split_ratio: float = 0.20
    poison_ratio: float = 0.10             # 10% label flipping (Jailbreak -> Benign)
    poison_detection_threshold: float = 0.65  # Threshold for Self-Filtering Sanitizer

    # Defense Thresholds
    cascade_low_threshold: float = 0.15     # Safe boundary
    cascade_high_threshold: float = 0.85    # Jailbreak boundary
    guard_decision_threshold: float = 0.50  # Final Gatekeeper binary threshold
    lookahead_band_min: float = 0.40        # Gray-zone lower bound for 10-token lookahead
    lookahead_band_max: float = 0.70        # Gray-zone upper bound for 10-token lookahead
    lookahead_tokens: int = 10

    # Training Hyperparameters
    encoder_epochs: int = 2
    encoder_lr: float = 2e-5
    encoder_batch_size: int = 16
    encoder_max_length: int = 256

    vlm_epochs: int = 1
    vlm_lr: float = 1e-4
    vlm_batch_size: int = 2
    vlm_gradient_accumulation_steps: int = 4

    # Sample caps (for quick_test)
    quick_test_samples_per_drop: int = 30
    full_run_samples_per_drop: int = 200

    # Sub-configs
    hardware: HardwareConfig = field(default_factory=HardwareConfig)
    vlm_params: VLMConfig = field(default_factory=VLMConfig)
    lora_params: LoRAConfig = field(default_factory=LoRAConfig)

    # Google Drive Backup
    drive_backup_dir: Optional[Path] = None

    def __post_init__(self):
        # Auto-detect Drive backup directory
        if self.drive_backup_dir is None:
            self.drive_backup_dir = detect_colab_drive_backup_dir()

        # Adjust batch size and precision for CPU if CUDA is not available
        if self.hardware.device == "cpu":
            self.encoder_batch_size = 4
            self.vlm_batch_size = 1
            self.vlm_gradient_accumulation_steps = 1
            self.vlm_params.load_in_4bit = False
            self.vlm_params.bnb_4bit_compute_dtype = "float32"
            if not self.quick_test:
                self.quick_test = True  # Auto-fallback to quick_test on CPU to prevent freeze


def get_default_config() -> ExperimentConfig:
    return ExperimentConfig()

