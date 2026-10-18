"""
src/evaluator.py - Metric Evaluation, Table Reporting & ROC Visualization
RMIT HACKATHON 2026 - Multimodal Gatekeeper AI Safety Benchmark
"""

import os
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix
)

from config import RESULTS_DIR
from src.utils import sync_to_drive

logger = logging.getLogger("Gatekeeper")


# -----------------------------------------------------------------------------
# 1. Core Metric Computation
# -----------------------------------------------------------------------------

def compute_metrics(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    hard_benign_mask: Optional[np.ndarray] = None,
    threshold: float = 0.50
) -> Dict[str, float]:
    """
    Computes comprehensive AI Safety evaluation metrics:
    - ROC-AUC
    - F1-Score, Precision, Recall (Jailbreak Detection Rate)
    - Over-refusal Rate (FPR specifically on Hard Benign queries)
    """
    y_pred = (y_probs >= threshold).astype(int)
    
    try:
        auc = float(roc_auc_score(y_true, y_probs))
    except Exception:
        auc = 0.50

    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    precision = float(precision_score(y_true, y_pred, zero_division=0))
    recall = float(recall_score(y_true, y_pred, zero_division=0))
    
    # Over-refusal Rate: False Positive Rate on Hard Benign subset
    if hard_benign_mask is not None and hard_benign_mask.sum() > 0:
        hb_preds = y_pred[hard_benign_mask]
        over_refusal_rate = float(hb_preds.sum() / len(hb_preds))
    else:
        # Standard FPR across all benign samples
        benign_mask = y_true == 0
        if benign_mask.sum() > 0:
            over_refusal_rate = float(y_pred[benign_mask].sum() / benign_mask.sum())
        else:
            over_refusal_rate = 0.0

    return {
        "ROC_AUC": auc,
        "F1_Score": f1,
        "Precision": precision,
        "Recall": recall,
        "Over_refusal_Rate": over_refusal_rate
    }


# -----------------------------------------------------------------------------
# 2. Benchmark Reporter
# -----------------------------------------------------------------------------

class BenchmarkEvaluator:
    def __init__(self, results_dir: Path = RESULTS_DIR, drive_backup_dir: Optional[Path] = None):
        self.results_dir = results_dir
        self.drive_backup_dir = drive_backup_dir
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def print_markdown_table(self, title: str, df: pd.DataFrame) -> None:
        """Prints a styled Markdown table to the console with cross-platform encoding safety."""
        banner = "=" * 80
        header_text = f"\n{banner}\n {title}\n{banner}"
        try:
            print(header_text)
        except UnicodeEncodeError:
            print(header_text.encode("ascii", errors="replace").decode("ascii"))

        try:
            output = df.to_markdown(index=False)
        except Exception:
            output = df.to_string(index=False)

        try:
            print(output)
            print(banner + "\n")
        except UnicodeEncodeError:
            print(output.encode("ascii", errors="replace").decode("ascii"))
            print(banner + "\n")

    def save_and_sync_table(self, filename: str, df: pd.DataFrame) -> Path:
        """Saves DataFrame as CSV in results/ and syncs with Google Drive."""
        out_path = self.results_dir / filename
        df.to_csv(out_path, index=False)
        sync_to_drive(out_path, self.drive_backup_dir)
        return out_path

    # -------------------------------------------------------------------------
    # Table 0: Poison Self-Filtering Evaluation
    # -------------------------------------------------------------------------
    def report_table0_poison_filtering(
        self,
        filter_metrics: Dict[str, float],
        auc_noisy_train: float,
        auc_cleaned_train: float
    ) -> pd.DataFrame:
        data = [{
            "Metric": "Poison Detection Precision",
            "Value": f"{filter_metrics['poison_precision']:.2%}"
        }, {
            "Metric": "Poison Detection Recall",
            "Value": f"{filter_metrics['poison_recall']:.2%}"
        }, {
            "Metric": "Poison Detection F1-Score",
            "Value": f"{filter_metrics['poison_f1']:.4f}"
        }, {
            "Metric": "Poisoned Samples Cleaned",
            "Value": f"{filter_metrics['total_poison_caught']} / {filter_metrics['total_poison_injected']}"
        }, {
            "Metric": "VLM ROC-AUC (Trained on Noisy Data)",
            "Value": f"{auc_noisy_train:.4f}"
        }, {
            "Metric": "VLM ROC-AUC (Trained on Cleaned Data)",
            "Value": f"{auc_cleaned_train:.4f}"
        }, {
            "Metric": "ROC-AUC Delta (Gain from Sanitization)",
            "Value": f"+{auc_cleaned_train - auc_noisy_train:.4f}"
        }]
        df = pd.DataFrame(data)
        self.print_markdown_table("TABLE 0: SELF-FILTERING POISON SANITIZER REPORT", df)
        self.save_and_sync_table("table0_poison_filtering.csv", df)
        return df

    # -------------------------------------------------------------------------
    # Table 1: Method Comparison
    # -------------------------------------------------------------------------
    def report_table1_method_comparison(
        self,
        methods_data: List[Dict[str, Any]]
    ) -> pd.DataFrame:
        """
        methods_data expects list of dicts with keys:
        [Method, ROC_AUC, F1_Score, Recall, Over_refusal_Rate, Latency_ms, Peak_VRAM_MB]
        """
        df = pd.DataFrame(methods_data)
        self.print_markdown_table("TABLE 1: COMPREHENSIVE METHOD COMPARISON", df)
        self.save_and_sync_table("table1_method_comparison.csv", df)
        return df

    # -------------------------------------------------------------------------
    # Table 2: Continual Rounds & Novel Attack (Leave-One-Drop-Out)
    # -------------------------------------------------------------------------
    def report_table2_continual_and_novel(
        self,
        continual_rounds_data: List[Dict[str, Any]],
        novel_attack_data: Dict[str, Any]
    ) -> pd.DataFrame:
        all_rows = list(continual_rounds_data)
        all_rows.append(novel_attack_data)
        df = pd.DataFrame(all_rows)
        self.print_markdown_table("TABLE 2: CONTINUAL ROUNDS & NOVEL ATTACK GENERALIZATION", df)
        self.save_and_sync_table("table2_continual_and_novel.csv", df)
        return df

    # -------------------------------------------------------------------------
    # Table 3: End-to-End Protection for Target Model
    # -------------------------------------------------------------------------
    def report_table3_end_to_end_target(
        self,
        e2e_results: Dict[str, Dict[str, float]]
    ) -> pd.DataFrame:
        rows = []
        for config_name, metrics in e2e_results.items():
            rows.append({
                "Defense Configuration": config_name,
                "Attack Success Rate (ASR) [lower is better]": f"{metrics['ASR']:.2%}",
                "Over-refusal Rate (FPR) [lower is better]": f"{metrics['Over_refusal_Rate']:.2%}",
                "End-to-End Latency (ms)": f"{metrics['Latency_ms']:.1f} ms"
            })
        df = pd.DataFrame(rows)
        self.print_markdown_table("TABLE 3: TARGET MODEL (QWEN/QWEN3.5-4B) END-TO-END DEFENSE", df)
        self.save_and_sync_table("table3_end_to_end_4b.csv", df)
        self.save_and_sync_table("table3_end_to_end_target.csv", df)
        return df

    # -------------------------------------------------------------------------
    # 4. ROC Curve Visualization
    # -------------------------------------------------------------------------
    def plot_roc_curves(
        self,
        y_true: np.ndarray,
        predictions_dict: Dict[str, np.ndarray],
        output_filename: str = "roc_curves.png"
    ) -> Path:
        """Generates and saves publication-quality ROC Curve comparison plot."""
        plt.figure(figsize=(9, 7))
        sns.set_theme(style="whitegrid")

        colors = ["#2b5c8f", "#d95f02", "#7570b3", "#e7298a", "#1b9e77"]
        
        for idx, (name, probs) in enumerate(predictions_dict.items()):
            try:
                fpr, tpr, _ = roc_curve(y_true, probs)
                auc_val = roc_auc_score(y_true, probs)
                color = colors[idx % len(colors)]
                plt.plot(fpr, tpr, label=f"{name} (AUC = {auc_val:.3f})", color=color, linewidth=2.2)
            except Exception as e:
                logger.warning(f"Could not compute ROC curve for {name}: {e}")

        plt.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random Guess (AUC = 0.500)")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate (Over-refusal)", fontsize=12)
        plt.ylabel("True Positive Rate (Detection Rate)", fontsize=12)
        plt.title("ROC-AUC Comparison: Multimodal Gatekeeper Defenses", fontsize=14, fontweight="bold")
        plt.legend(loc="lower right", frameon=True, fontsize=10)
        plt.tight_layout()

        out_path = self.results_dir / output_filename
        plt.savefig(out_path, dpi=300)
        plt.close()
        sync_to_drive(out_path, self.drive_backup_dir)
        logger.info(f"[Evaluator] Saved ROC curves plot to {out_path.name}")
        return out_path
