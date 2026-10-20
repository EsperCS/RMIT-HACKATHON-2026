"""
src/utils.py - System Utilities: Seed, Anti-OOM, Auto-Resume State, Hardware & Drive Sync
RMIT HACKATHON 2026 - Multimodal Gatekeeper AI Safety Benchmark
"""

import os
import gc
import sys
import json
import random
import shutil
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
import numpy as np
import torch

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import BASE_DIR, SYNTHETIC_IMAGES_DIR, DOWNLOADS_DIR, CHECKPOINTS_DIR, RESULTS_DIR, HardwareConfig


def resolve_image_path(img_path: Optional[str]) -> Optional[Path]:
    """
    Cross-platform image path resolver.
    Handles relative paths, absolute paths from another machine, and filename-only lookups.
    """
    if not img_path:
        return None
    p = Path(img_path)
    if p.is_file() and p.exists():
        return p
    # Relative to repository base directory
    cand1 = BASE_DIR / p
    if cand1.is_file() and cand1.exists():
        return cand1
    # Check by filename inside synthetic_images/
    cand2 = SYNTHETIC_IMAGES_DIR / p.name
    if cand2.is_file() and cand2.exists():
        return cand2
    # Check by filename inside downloads/
    cand3 = DOWNLOADS_DIR / p.name
    if cand3.is_file() and cand3.exists():
        return cand3
    return None

# Setup unbuffered real-time logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("Gatekeeper")


def seed_everything(seed: int = 42) -> None:
    """
    Enforces strict reproducibility across Python, NumPy, and PyTorch.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    logger.info(f"[Reproducibility] Global seed fixed to: {seed}")


def get_hardware_status() -> HardwareConfig:
    """
    Inspects available compute hardware (CUDA/CPU), logs status, and warns if running on CPU.
    """
    hw = HardwareConfig()
    if torch.cuda.is_available():
        logger.info(
            f"[Hardware] CUDA GPU Detected: {hw.gpu_name} | VRAM: {hw.vram_gb:.2f} GB "
            f"| Colab T4: {hw.is_colab_t4} | Colab L4: {hw.is_colab_l4}"
        )
    else:
        logger.warning(
            "[Hardware WARNING] No CUDA GPU detected! Running on CPU. "
            "Google Colab users: please navigate to Runtime -> Change runtime type -> Select T4 GPU!"
        )
    return hw


def cleanup_vram() -> None:
    """
    Aggressive VRAM garbage collection to prevent Out-Of-Memory (OOM) between stages.
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        allocated_mb = torch.cuda.memory_allocated() / (1024 ** 2)
        reserved_mb = torch.cuda.memory_reserved() / (1024 ** 2)
        logger.debug(f"[VRAM Cleanup] Allocated: {allocated_mb:.1f} MB | Reserved: {reserved_mb:.1f} MB")


def get_peak_vram_mb() -> float:
    """
    Returns the peak GPU memory allocated in MB.
    """
    if torch.cuda.is_available():
        return float(torch.cuda.max_memory_allocated() / (1024 ** 2))
    return 0.0


def reset_peak_vram() -> None:
    """
    Resets peak memory stats for accurate per-module profiling.
    """
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def sync_to_drive(source_path: Path, drive_backup_dir: Optional[Path]) -> None:
    """
    Safely mirrors a local file or directory to Google Drive for crash/disconnect resilience.
    """
    if drive_backup_dir is None:
        return
    try:
        drive_backup_dir.mkdir(parents=True, exist_ok=True)
        rel_path = source_path.name
        dest_path = drive_backup_dir / rel_path

        if source_path.is_file():
            shutil.copy2(source_path, dest_path)
            logger.debug(f"[Drive Sync] Mirrored file {source_path.name} -> {dest_path}")
        elif source_path.is_dir():
            shutil.copytree(source_path, dest_path, dirs_exist_ok=True)
            logger.debug(f"[Drive Sync] Mirrored dir {source_path.name} -> {dest_path}")
    except Exception as e:
        logger.warning(f"[Drive Sync Warning] Could not mirror {source_path} to Drive: {e}")


def resolve_model_id(preferred_id: str, fallback_id: str) -> str:
    """
    Verifies if a Hugging Face model repository is accessible.
    If the preferred repo does not exist or requires authorization, falls back seamlessly.
    """
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        api.model_info(preferred_id, timeout=3.0)
        logger.info(f"[Model Resolver] Verified access to preferred model: '{preferred_id}'")
        return preferred_id
    except Exception:
        logger.warning(
            f"[Model Resolver] Could not access '{preferred_id}'. "
            f"Falling back safely to '{fallback_id}'"
        )
        return fallback_id


class ExperimentState:
    """
    Atomic Experiment State Manager for 1-click Auto-Resume and recovery.
    Maintains `experiment_state.json` and syncs with Google Drive.
    """
    def __init__(self, checkpoints_dir: Path = CHECKPOINTS_DIR, drive_backup_dir: Optional[Path] = None):
        self.checkpoints_dir = checkpoints_dir
        self.drive_backup_dir = drive_backup_dir
        self.state_file = self.checkpoints_dir / "experiment_state.json"
        self.state_data: Dict[str, Any] = self._load_state()

    def _load_state(self) -> Dict[str, Any]:
        # 1. Try loading from local checkpoints
        if self.state_file.exists():
            try:
                with open(self.state_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    logger.info(f"[State Manager] Loaded existing local state: {len(data.get('completed_steps', []))} steps completed.")
                    return data
            except Exception as e:
                logger.warning(f"[State Manager] Corrupted local state file, attempting recovery: {e}")

        # 2. Try recovering from Google Drive
        if self.drive_backup_dir:
            drive_state = self.drive_backup_dir / "experiment_state.json"
            if drive_state.exists():
                try:
                    with open(drive_state, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        logger.info(f"[State Manager] Successfully recovered state from Google Drive!")
                        # Copy back to local
                        with open(self.state_file, "w", encoding="utf-8") as f_out:
                            json.dump(data, f_out, indent=2)
                        return data
                except Exception as e:
                    logger.warning(f"[State Manager] Drive state recovery failed: {e}")

        # Default fresh state
        return {
            "completed_steps": [],
            "step_metadata": {},
            "metrics": {}
        }

    def is_completed(self, step_name: str) -> bool:
        return step_name in self.state_data.get("completed_steps", [])

    def mark_completed(self, step_name: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        if step_name not in self.state_data["completed_steps"]:
            self.state_data["completed_steps"].append(step_name)
        if metadata:
            self.state_data["step_metadata"][step_name] = metadata
        self._atomic_save()
        logger.info(f"[State Manager] Step '{step_name}' marked as COMPLETED.")

    def record_metric(self, key: str, value: Any) -> None:
        self.state_data["metrics"][key] = value
        self._atomic_save()

    def _atomic_save(self) -> None:
        """
        Writes state to temporary file first, then atomically replaces to prevent corruption.
        """
        temp_file = self.state_file.with_suffix(".tmp")
        try:
            with open(temp_file, "w", encoding="utf-8") as f:
                json.dump(self.state_data, f, indent=2)
            temp_file.replace(self.state_file)
            # Sync to Drive
            sync_to_drive(self.state_file, self.drive_backup_dir)
        except Exception as e:
            logger.error(f"[State Manager] Error saving state: {e}")

    def save_intermediate(self, filename: str, content: Any) -> Path:
        """
        Saves intermediate predictions (.parquet / .npy / .json) and syncs to Drive.
        """
        target_path = self.checkpoints_dir / filename
        if filename.endswith(".npy"):
            np.save(target_path, content)
        elif filename.endswith(".json"):
            with open(target_path, "w", encoding="utf-8") as f:
                json.dump(content, f, indent=2)
        elif hasattr(content, "to_parquet"):
            content.to_parquet(target_path, index=False)
        else:
            raise ValueError(f"Unsupported intermediate save format for: {filename}")

        sync_to_drive(target_path, self.drive_backup_dir)
        return target_path

    def load_intermediate(self, filename: str, allow_missing: bool = False) -> Any:
        target_path = self.checkpoints_dir / filename
        if not target_path.exists() and self.drive_backup_dir:
            drive_path = self.drive_backup_dir / filename
            if drive_path.exists():
                shutil.copy2(drive_path, target_path)

        if not target_path.exists():
            if allow_missing:
                return None
            raise FileNotFoundError(f"Intermediate file not found: {filename}")

        if filename.endswith(".npy"):
            return np.load(target_path, allow_pickle=True)
        elif filename.endswith(".json"):
            with open(target_path, "r", encoding="utf-8") as f:
                return json.load(f)
        elif filename.endswith(".parquet"):
            import pandas as pd
            return pd.read_parquet(target_path)
        return None


if __name__ == "__main__":
    print("=" * 70)
    print(" [DEBUG TOOL] src/utils.py - Diagnostics & System Utilities Test")
    print("=" * 70)
    
    # 1. Test Seed reproducibility
    seed_everything(42)
    val_rand = random.random()
    val_np = np.random.rand()
    val_torch = torch.rand(1).item()
    print(f"[Seed Check] Python: {val_rand:.4f} | NumPy: {val_np:.4f} | PyTorch: {val_torch:.4f}")
    
    # 2. Test Hardware Status & VRAM Cleanup
    hw = get_hardware_status()
    print(f"[Hardware] Device: {hw.device} | Name: {hw.gpu_name} | VRAM: {hw.vram_gb:.2f} GB")
    cleanup_vram()
    print(f"[VRAM Cleanup] Current allocated: {torch.cuda.memory_allocated() / (1024**2):.1f} MB" if torch.cuda.is_available() else "[VRAM Cleanup] CPU mode OK.")
    
    # 3. Test State Manager & Intermediate Save/Load
    test_state_dir = CHECKPOINTS_DIR / "_debug_temp_state"
    test_state_dir.mkdir(parents=True, exist_ok=True)
    state_mgr = ExperimentState(test_state_dir)
    state_mgr.mark_completed("debug_test_step", {"timestamp": "2026-10-19", "status": "passed"})
    is_done = state_mgr.is_completed("debug_test_step")
    print(f"[State Manager] 'debug_test_step' completed: {is_done} (Expected: True)")
    
    # Test intermediate save/load
    test_arr = np.array([0.1, 0.5, 0.9], dtype=np.float32)
    saved_file = state_mgr.save_intermediate("test_probs.npy", test_arr)
    loaded_arr = state_mgr.load_intermediate("test_probs.npy")
    np.testing.assert_array_almost_equal(test_arr, loaded_arr)
    print(f"[Intermediate I/O] Saved & Loaded array match: OK ({saved_file.name})")
    
    # Clean up temp state
    shutil.rmtree(test_state_dir, ignore_errors=True)
    
    # 4. Test resolve_image_path
    resolved = resolve_image_path("non_existent_image.png")
    print(f"[Path Resolver] Non-existent fallback check: {resolved} (Expected: None)")
    
    print("=" * 70)
    print(" [SUCCESS] src/utils.py utilities and state manager passed all tests.")
    print("=" * 70)

