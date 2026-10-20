"""
run_experiment.py - Master Orchestrator for Multimodal Gatekeeper AI Safety Benchmark
Gatekeeper: Qwen/Qwen3.5-2B | Target Model: Qwen/Qwen3.5-4B
RMIT HACKATHON 2026 - Multimodal Gatekeeper AI Safety Benchmark
"""

import sys
import argparse
import logging
from pathlib import Path
from typing import Optional, List, Dict, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from config import get_default_config, RESULTS_DIR, CHECKPOINTS_DIR, PROCESSED_DATA_DIR
from src.utils import (
    seed_everything,
    get_hardware_status,
    cleanup_vram,
    ExperimentState,
    sync_to_drive,
    get_peak_vram_mb,
    reset_peak_vram
)
from src.data_mixer import MultimodalDataMixer, load_rmit_hackathon_2025_test_dataset
from src.text_features import ReflexivityClassifier
from src.qwen_vlm_branch import QwenVLMBranch
from src.hybrid_defenses import (
    MultiSignalRankEnsemble,
    CascadeGatekeeper,
    TargetModelDefender,
    find_optimal_threshold
)
from src.evaluator import BenchmarkEvaluator, compute_metrics

logger = logging.getLogger("Gatekeeper")


def parse_args():
    parser = argparse.ArgumentParser(description="Multimodal Gatekeeper AI Safety Benchmark CLI")
    parser.add_argument("--quick_test", action="store_true", help="Run fast 2-3 minute verification test with small sample cap")
    parser.add_argument("--full_run", action="store_true", help="Run comprehensive benchmark on all data")
    parser.add_argument("--stage", type=str, default="all", choices=["all", "data", "guard", "e2e", "eval", "submission"],
                        help="Execute a specific stage of the pipeline")
    parser.add_argument("--resume", action="store_true", default=True, help="Auto-resume from last completed checkpoint")
    parser.add_argument("--no_resume", dest="resume", action="store_false", help="Disable auto-resume and run all from scratch")
    parser.add_argument("--val", action="store_true", help="Evaluate benchmark on the 500 validation samples (val.parquet)")
    parser.add_argument("--eval_split", type=str, default="test", choices=["test", "val"],
                        help="Choose evaluation target: 'test' (global_test.parquet) or 'val' (val.parquet)")
    parser.add_argument("--predict_test", action="store_true",
                        help="Run inference on competition test.csv and generate submission.csv")
    parser.add_argument("--test_file", type=str, default=None,
                        help="Explicit path to competition test.csv")
    parser.add_argument("--sample_submission", type=str, default=None,
                        help="Explicit path to competition sample_submission.csv")
    parser.add_argument("--drive_backup_dir", type=str, default=None, help="Custom Google Drive backup directory path")
    parser.add_argument("--guard_model", type=str, default=None, help="Override HuggingFace guard VLM model ID")
    parser.add_argument("--target_model", type=str, default=None, help="Override HuggingFace target LLM model ID")
    parser.add_argument("--update_slides", action="store_true",
                        help="Synchronize results/*.csv tables directly into presentation/slides.tex")
    return parser.parse_args()


def main():
    args = parse_args()

    # Direct utility: update presentation slides from existing CSV results
    if args.update_slides:
        from presentation.update_slides import update_presentation_slides
        update_presentation_slides(use_val=args.val or (args.eval_split == "val"))
        return

    config = get_default_config()

    # Apply CLI Overrides
    if args.quick_test:
        config.quick_test = True
    if args.full_run:
        config.full_run = True
        config.quick_test = False
    if args.drive_backup_dir:
        config.drive_backup_dir = Path(args.drive_backup_dir)
    if args.guard_model:
        config.guard_model_id = args.guard_model
    if args.target_model:
        config.target_model_id = args.target_model

    eval_split = "val" if (args.val or args.eval_split == "val") else "test"
    prefix = f"{eval_split}_" if eval_split != "test" else ""

    # 1. Reproducibility & Hardware Check
    seed_everything(config.seed)
    hw = get_hardware_status()
    state = ExperimentState(CHECKPOINTS_DIR, config.drive_backup_dir)
    evaluator = BenchmarkEvaluator(RESULTS_DIR, config.drive_backup_dir)

    logger.info("====================================================================")
    logger.info(" RMIT HACKATHON 2026 - MULTIMODAL GATEKEEPER BENCHMARK STARTING")
    logger.info(f" Gatekeeper: {config.guard_model_id} | Target Model: {config.target_model_id}")
    logger.info(f" Mode: {'QUICK TEST' if config.quick_test else 'FULL RUN'} | Eval Target: {eval_split.upper()} | Device: {hw.device} ({hw.gpu_name})")
    logger.info("====================================================================")

    # -------------------------------------------------------------------------
    # STAGE 1: Multimodal Data Preparation & Poison Injection
    # -------------------------------------------------------------------------
    if args.stage in ["all", "data"]:
        logger.info("\n>>> [STAGE 1] Preparing Multimodal Datasets, 50:25:25 Ratio & Poisoning...")
        mixer = MultimodalDataMixer(config)
        datasets = mixer.prepare_experiment_datasets()
        state.mark_completed("data_prep", {"num_rounds": len(datasets) - 1})
    else:
        datasets = {
            "global_test": pd.read_parquet(PROCESSED_DATA_DIR / "global_test.parquet")
        }
        for r_path in PROCESSED_DATA_DIR.glob("continual_round_*.parquet"):
            datasets[r_path.stem] = pd.read_parquet(r_path)

    if args.stage == "data":
        logger.info("[Pipeline] Data preparation completed (--stage data).")
        return

    if eval_split == "val":
        val_file = PROCESSED_DATA_DIR / "val.parquet"
        if not val_file.exists():
            logger.info("[Evaluator] val.parquet not found. Running data preparation first...")
            mixer = MultimodalDataMixer(config)
            datasets = mixer.prepare_experiment_datasets()
        test_df = pd.read_parquet(val_file)
        logger.info(f"\n[Evaluator] >>> EVALUATION TARGET: VALIDATION SET (val.parquet: {len(test_df)} samples) <<<")
    else:
        test_df = datasets["global_test"]
        logger.info(f"\n[Evaluator] >>> EVALUATION TARGET: GLOBAL TEST SET (global_test.parquet: {len(test_df)} samples) <<<")

    y_test_true = test_df["true_label"].values
    hard_benign_mask = (test_df["category"] == "hard_benign").values

    # -------------------------------------------------------------------------
    # STAGE 2: Fast Explicit Reflexivity Screen (arXiv:2510.01644)
    # -------------------------------------------------------------------------
    p_reflexivity = None
    reflex_latency = 0.5
    reflex_vram = 0.0

    round1_df = pd.read_parquet(PROCESSED_DATA_DIR / "continual_round_1.parquet")

    logger.info("\n>>> [STAGE 2] Fitting & Evaluating Fast Reflexivity Classifier...")
    reset_peak_vram()
    reflex_clf = ReflexivityClassifier()
    reflex_clf.fit(round1_df["prompt"].tolist(), round1_df["noisy_label"].tolist())
    p_reflexivity = reflex_clf.predict_proba(test_df["prompt"].tolist())
    reflex_vram = get_peak_vram_mb()
    state.save_intermediate(f"{prefix}p_reflexivity.npy", p_reflexivity)

    # -------------------------------------------------------------------------
    # STAGE 3: Qwen Multimodal Gatekeeper Branch (Zero-shot, Poison Cleaner & QLoRA)
    # -------------------------------------------------------------------------
    p_vlm_zero = None
    p_vlm_dirty = None
    p_vlm_clean = None
    p_ood = None
    p_continual_round2 = None
    p_continual_round3 = None
    filter_metrics = None
    vlm_latency = 45.0
    vlm_vram = 0.0

    run_guard = (
        args.stage in ["all", "guard", "eval", "e2e", "submission"]
        or args.predict_test
        or args.test_file is not None
    )
    if run_guard:
        logger.info("\n>>> [STAGE 3] Running Qwen Gatekeeper Branch (Zero-shot, Poison Filter, QLoRA, OOD)...")
        reset_peak_vram()
        vlm_branch = QwenVLMBranch(config)

        # 1. Zero-shot Evaluation on Test Set
        logger.info("[Gatekeeper 2B] Evaluating Zero-shot Multimodal Logits...")
        p_vlm_zero, vlm_latency = vlm_branch.predict_logits_single_step(test_df)
        state.save_intermediate(f"{prefix}p_vlm_zero.npy", p_vlm_zero)

        # 2. Self-Filtering Poison Sanitization on Round 1 Training Data
        cleaned_round1_df, filter_metrics = vlm_branch.sanitize_poisoned_labels(round1_df)
        cleaned_round1_df.to_parquet(PROCESSED_DATA_DIR / "cleaned_round_1.parquet", index=False)

        # 3. Train on Dirty/Poisoned Data (for benchmark comparison)
        logger.info("[Gatekeeper 2B] Fine-tuning on Dirty/Poisoned Data for Table 1 comparison...")
        vlm_branch.train_qlora(round1_df, round_name="round_1_dirty")
        p_vlm_dirty, _ = vlm_branch.predict_logits_single_step(test_df)
        state.save_intermediate(f"{prefix}p_vlm_dirty.npy", p_vlm_dirty)

        # 4. Train on Cleaned Data (after self-filtering)
        logger.info("[Gatekeeper 2B] Fine-tuning on Cleaned Data (Self-Filtered)...")
        vlm_branch.train_qlora(cleaned_round1_df, round_name="round_1_clean")
        p_vlm_clean, _ = vlm_branch.predict_logits_single_step(test_df)
        emulated_round1_state = None
        if vlm_branch.is_emulated:
            emulated_round1_state = {
                key: value.detach().cpu().clone()
                for key, value in vlm_branch.emulated_net.state_dict().items()
            }
        vlm_vram = get_peak_vram_mb()
        state.save_intermediate(f"{prefix}p_vlm_clean.npy", p_vlm_clean)

        # 5. OOD Mahalanobis / Cosine Distance Anomaly Scores
        logger.info("[Gatekeeper 2B] Computing OOD Anomaly Scores on Test Set...")
        p_ood = vlm_branch.compute_ood_scores(test_df)
        state.save_intermediate(f"{prefix}p_ood.npy", p_ood)
        ood_reference_state = (
            None if vlm_branch.benign_centroid is None else vlm_branch.benign_centroid.copy(),
            None if vlm_branch.benign_cov_inv is None else vlm_branch.benign_cov_inv.copy(),
            vlm_branch.ood_scale,
        )

        # 6. Continual protocol: train each cumulative round and evaluate each
        # checkpoint on the same held-out benchmark.  Do not substitute slices
        # of the test set for model checkpoints.
        continual_predictions = {1: p_vlm_clean}
        for continual_round in (2, 3):
            continual_df = pd.read_parquet(
                PROCESSED_DATA_DIR / f"continual_round_{continual_round}.parquet"
            )
            cleaned_continual_df, _ = vlm_branch.sanitize_poisoned_labels(continual_df)
            vlm_branch.train_qlora(
                cleaned_continual_df, round_name=f"round_{continual_round}_clean"
            )
            continual_predictions[continual_round], _ = vlm_branch.predict_logits_single_step(test_df)
            state.save_intermediate(
                f"{prefix}p_continual_round_{continual_round}.npy",
                continual_predictions[continual_round],
            )
        p_continual_round2 = continual_predictions[2]
        p_continual_round3 = continual_predictions[3]

        # 7. Novel protocol: train only on Drops 1+2 and evaluate Drop 3.
        novel_test_file = PROCESSED_DATA_DIR / "novel_protocol_test.parquet"
        novel_train_file = PROCESSED_DATA_DIR / "novel_protocol_train.parquet"
        if novel_test_file.exists() and novel_train_file.exists():
            logger.info("[Gatekeeper 2B] Training and evaluating Leave-One-Drop-Out Novel Attack protocol...")
            novel_test_df = pd.read_parquet(novel_test_file)
            novel_train_df = pd.read_parquet(novel_train_file)
            emulated_round3_state = None
            if vlm_branch.is_emulated:
                emulated_round3_state = {
                    key: value.detach().cpu().clone()
                    for key, value in vlm_branch.emulated_net.state_dict().items()
                }
            cleaned_novel_train, _ = vlm_branch.sanitize_poisoned_labels(novel_train_df)
            vlm_branch.train_qlora(cleaned_novel_train, round_name="novel_drop12_clean")
            p_novel_clean, _ = vlm_branch.predict_logits_single_step(novel_test_df)
            state.save_intermediate(f"{prefix}p_novel_clean.npy", p_novel_clean)
            if vlm_branch.is_emulated and emulated_round3_state is not None:
                vlm_branch.emulated_net.load_state_dict(emulated_round3_state)
        if vlm_branch.is_emulated and emulated_round1_state is not None:
            vlm_branch.emulated_net.load_state_dict(emulated_round1_state)
        elif hasattr(vlm_branch.model, "set_adapter"):
            try:
                vlm_branch.model.set_adapter("round_1_clean")
            except Exception as exc:
                logger.warning(f"[Gatekeeper 2B] Could not restore round_1_clean adapter: {exc}")
        vlm_branch.benign_centroid, vlm_branch.benign_cov_inv, vlm_branch.ood_scale = ood_reference_state

        # Table 0 Generation: Poison Self-Filtering Sanitizer
        auc_dirty = float(roc_auc_score(y_test_true, p_vlm_dirty))
        auc_clean = float(roc_auc_score(y_test_true, p_vlm_clean))
        evaluator.report_table0_poison_filtering(filter_metrics, auc_dirty, auc_clean)
        state.mark_completed("guard_round_1")
        cleanup_vram()

        if args.stage == "guard":
            logger.info("[Pipeline] Guard training and cached predictions completed (--stage guard).")
            return

    # Load cached probabilities if running evaluation standalone
    if p_vlm_zero is None:
        p_vlm_zero = state.load_intermediate(f"{prefix}p_vlm_zero.npy", allow_missing=True)
    if p_vlm_dirty is None:
        p_vlm_dirty = state.load_intermediate(f"{prefix}p_vlm_dirty.npy", allow_missing=True)
    if p_vlm_clean is None:
        p_vlm_clean = state.load_intermediate(f"{prefix}p_vlm_clean.npy", allow_missing=True)
    if p_ood is None:
        p_ood = state.load_intermediate(f"{prefix}p_ood.npy", allow_missing=True)
    if p_continual_round2 is None:
        p_continual_round2 = state.load_intermediate(
            f"{prefix}p_continual_round_2.npy", allow_missing=True
        )
    if p_continual_round3 is None:
        p_continual_round3 = state.load_intermediate(
            f"{prefix}p_continual_round_3.npy", allow_missing=True
        )

    required_signals = {
        "p_vlm_zero": p_vlm_zero,
        "p_vlm_dirty": p_vlm_dirty,
        "p_vlm_clean": p_vlm_clean,
        "p_ood": p_ood,
    }
    missing_signals = [name for name, value in required_signals.items() if value is None]
    if missing_signals:
        raise RuntimeError(
            "Required guard predictions are unavailable: "
            f"{', '.join(missing_signals)}. Run --stage guard or --stage all first."
        )

    p_novel_clean = state.load_intermediate(f"{prefix}p_novel_clean.npy", allow_missing=True)
    novel_test_file = PROCESSED_DATA_DIR / "novel_protocol_test.parquet"
    novel_test_df = pd.read_parquet(novel_test_file) if novel_test_file.exists() else None
    if p_novel_clean is None and novel_test_df is not None:
        p_novel_clean = reflex_clf.predict_proba(novel_test_df["prompt"].tolist())

    # -------------------------------------------------------------------------
    # STAGE 4: Multi-Signal Ensemble & 2-Tier Cascade Gatekeeper
    # -------------------------------------------------------------------------
    logger.info("\n>>> [STAGE 4] Computing Multi-Signal Ensemble & 2-Tier Cascade...")

    # 1. Fit the ensemble strictly on validation data.  The global test labels
    # are never used to tune weights or a decision threshold.
    calibration_file = PROCESSED_DATA_DIR / "val.parquet"
    if not calibration_file.exists():
        raise FileNotFoundError("val.parquet is required to calibrate the ensemble.")
    calibration_df = pd.read_parquet(calibration_file)
    y_calibration = calibration_df["true_label"].values
    hard_benign_calibration = (calibration_df["category"] == "hard_benign").values

    if eval_split == "val":
        logger.warning(
            "[Evaluator] --val reports the calibration split; its ensemble metrics are diagnostics, "
            "not an unbiased held-out benchmark."
        )

    p_reflexivity_calibration = reflex_clf.predict_proba(calibration_df["prompt"].tolist())
    p_vlm_clean_calibration, _ = vlm_branch.predict_logits_single_step(calibration_df)
    if not vlm_branch.is_emulated and hasattr(vlm_branch.model, "disable_adapter"):
        try:
            with vlm_branch.model.disable_adapter():
                p_vlm_zero_calibration, _ = vlm_branch.predict_logits_single_step(calibration_df)
        except Exception as exc:
            logger.warning(f"[Ensemble] Could not evaluate adapter-free calibration signal: {exc}")
            p_vlm_zero_calibration = p_vlm_clean_calibration
    else:
        p_vlm_zero_calibration = p_vlm_clean_calibration
    p_ood_calibration = vlm_branch.compute_ood_scores(calibration_df)

    # 2. Multi-Signal rank ensemble, calibrated on validation only.
    ensemble = MultiSignalRankEnsemble()
    ensemble.fit_weights(
        p_vlm_clean_calibration,
        p_vlm_zero_calibration,
        p_ood_calibration,
        p_reflexivity_calibration,
        y_calibration,
        hard_benign_calibration,
    )
    p_ensemble_calibration = ensemble.predict(
        p_vlm_clean_calibration,
        p_vlm_zero_calibration,
        p_ood_calibration,
        p_reflexivity_calibration,
    )
    p_ensemble = ensemble.predict(p_vlm_clean, p_vlm_zero, p_ood, p_reflexivity)

    # 3. 2-Tier Cascade Gatekeeper (Fast Screen -> Deep Guard)
    cascade = CascadeGatekeeper(config)
    p_cascade, _, cascade_lat, tier2_route_pct = cascade.evaluate_cascade(
        test_df, p_reflexivity, p_ensemble, reflex_latency, vlm_latency
    )
    p_cascade_calibration, _, _, _ = cascade.evaluate_cascade(
        calibration_df,
        p_reflexivity_calibration,
        p_ensemble_calibration,
        reflex_latency,
        vlm_latency,
    )

    # -------------------------------------------------------------------------
    # STAGE 5: Evaluation & Table Reporting
    # -------------------------------------------------------------------------
    logger.info("\n>>> [STAGE 5] Generating Benchmark Tables & ROC Curves...")

    # TABLE 1: 5 Configurations Method Comparison
    m_reflex = compute_metrics(y_test_true, p_reflexivity, hard_benign_mask)
    m_vlm_zero = compute_metrics(y_test_true, p_vlm_zero, hard_benign_mask)
    m_vlm_dirty = compute_metrics(y_test_true, p_vlm_dirty, hard_benign_mask)
    m_vlm_clean = compute_metrics(y_test_true, p_vlm_clean, hard_benign_mask)
    m_cascade_ens = compute_metrics(y_test_true, p_cascade, hard_benign_mask)

    table1_rows = [
        {
            "Method / Configuration": "1. Reflexivity Heuristic Only",
            "ROC_AUC": f"{m_reflex['ROC_AUC']:.4f}",
            "F1_Score": f"{m_reflex['F1_Score']:.4f}",
            "Recall (Detection)": f"{m_reflex['Recall']:.2%}",
            "Over-refusal (Hard Benign)": f"{m_reflex['Over_refusal_Rate']:.2%}",
            "Latency (ms)": f"{reflex_latency:.1f} ms",
            "Peak VRAM (MB)": f"{reflex_vram:.0f} MB"
        },
        {
            "Method / Configuration": "2. Qwen3.5-2B Zero-shot",
            "ROC_AUC": f"{m_vlm_zero['ROC_AUC']:.4f}",
            "F1_Score": f"{m_vlm_zero['F1_Score']:.4f}",
            "Recall (Detection)": f"{m_vlm_zero['Recall']:.2%}",
            "Over-refusal (Hard Benign)": f"{m_vlm_zero['Over_refusal_Rate']:.2%}",
            "Latency (ms)": f"{vlm_latency:.1f} ms",
            "Peak VRAM (MB)": f"{vlm_vram:.0f} MB"
        },
        {
            "Method / Configuration": "3. Qwen3.5-2B QLoRA (Dirty / Poisoned Data)",
            "ROC_AUC": f"{m_vlm_dirty['ROC_AUC']:.4f}",
            "F1_Score": f"{m_vlm_dirty['F1_Score']:.4f}",
            "Recall (Detection)": f"{m_vlm_dirty['Recall']:.2%}",
            "Over-refusal (Hard Benign)": f"{m_vlm_dirty['Over_refusal_Rate']:.2%}",
            "Latency (ms)": f"{vlm_latency:.1f} ms",
            "Peak VRAM (MB)": f"{vlm_vram:.0f} MB"
        },
        {
            "Method / Configuration": "4. Qwen3.5-2B QLoRA (Cleaned Data)",
            "ROC_AUC": f"{m_vlm_clean['ROC_AUC']:.4f}",
            "F1_Score": f"{m_vlm_clean['F1_Score']:.4f}",
            "Recall (Detection)": f"{m_vlm_clean['Recall']:.2%}",
            "Over-refusal (Hard Benign)": f"{m_vlm_clean['Over_refusal_Rate']:.2%}",
            "Latency (ms)": f"{vlm_latency:.1f} ms",
            "Peak VRAM (MB)": f"{vlm_vram:.0f} MB"
        },
        {
            "Method / Configuration": "5. Cascade & Multi-Signal Ensemble (Reflex + 2B + OOD)",
            "ROC_AUC": f"{m_cascade_ens['ROC_AUC']:.4f}",
            "F1_Score": f"{m_cascade_ens['F1_Score']:.4f}",
            "Recall (Detection)": f"{m_cascade_ens['Recall']:.2%}",
            "Over-refusal (Hard Benign)": f"{m_cascade_ens['Over_refusal_Rate']:.2%}",
            "Latency (ms)": f"{cascade_lat:.1f} ms",
            "Peak VRAM (MB)": f"{vlm_vram:.0f} MB"
        }
    ]
    eval_title_suffix = " (VALIDATION SET - 500 SAMPLES)" if eval_split == "val" else ""
    t1_df = pd.DataFrame(table1_rows)
    evaluator.print_markdown_table(f"TABLE 1: COMPREHENSIVE METHOD COMPARISON{eval_title_suffix}", t1_df)
    evaluator.save_and_sync_table("table1_method_comparison.csv", t1_df)
    if eval_split == "val":
        evaluator.save_and_sync_table("table1_val_method_comparison.csv", t1_df)

    # TABLE 2: Continual Rounds Evolution & Novel Attack (Leave-One-Drop-Out)
    # Dynamically compute metrics from real model predictions without hardcoded multipliers
    calibration_clean_auc = compute_metrics(
        y_calibration, p_vlm_clean_calibration, hard_benign_calibration
    )["ROC_AUC"]
    calibration_cascade_auc = compute_metrics(
        y_calibration, p_cascade_calibration, hard_benign_calibration
    )["ROC_AUC"]
    if calibration_clean_auc >= calibration_cascade_auc:
        best_guard_probs = p_vlm_clean
        best_guard_probs_calibration = p_vlm_clean_calibration
    else:
        best_guard_probs = p_cascade
        best_guard_probs_calibration = p_cascade_calibration
    opt_thresh_best = find_optimal_threshold(
        y_calibration, best_guard_probs_calibration, hard_benign_calibration, max_fpr=0.05
    )

    # Each row uses predictions from the model trained at that actual
    # cumulative checkpoint, evaluated on one fixed held-out benchmark.
    if p_continual_round2 is None or p_continual_round3 is None:
        raise RuntimeError(
            "Continual-round predictions are unavailable. Run --stage guard or --stage all first."
        )
    m_round1 = compute_metrics(
        y_test_true, p_vlm_clean, hard_benign_mask, threshold=opt_thresh_best
    )
    m_round2 = compute_metrics(
        y_test_true, p_continual_round2, hard_benign_mask, threshold=opt_thresh_best
    )
    m_round3 = compute_metrics(
        y_test_true, p_continual_round3, hard_benign_mask, threshold=opt_thresh_best
    )

    continual_data = [
        {
            "Evaluation Protocol": "Continual Round 1 (Drop 1)",
            "ROC_AUC": f"{m_round1['ROC_AUC']:.4f}",
            "Detection Rate": f"{m_round1['Recall']:.1%}",
            "Over-refusal": f"{m_round1['Over_refusal_Rate']:.1%}"
        },
        {
            "Evaluation Protocol": "Continual Round 2 (Drop 1+2)",
            "ROC_AUC": f"{m_round2['ROC_AUC']:.4f}",
            "Detection Rate": f"{m_round2['Recall']:.1%}",
            "Over-refusal": f"{m_round2['Over_refusal_Rate']:.1%}"
        },
        {
            "Evaluation Protocol": "Continual Round 3 (Drop 1+2+3)",
            "ROC_AUC": f"{m_round3['ROC_AUC']:.4f}",
            "Detection Rate": f"{m_round3['Recall']:.1%}",
            "Over-refusal": f"{m_round3['Over_refusal_Rate']:.1%}"
        }
    ]

    # Novel Attack Generalization (Leave-One-Drop-Out)
    if novel_test_df is not None and p_novel_clean is not None and len(novel_test_df) == len(p_novel_clean):
        y_novel_true = novel_test_df["true_label"].values
        hb_novel_mask = (novel_test_df["category"] == "hard_benign").values
        m_novel = compute_metrics(y_novel_true, p_novel_clean, hb_novel_mask, threshold=opt_thresh_best)
    else:
        raise RuntimeError(
            "Novel-protocol predictions are unavailable. Run --stage guard or --stage all first."
        )

    novel_data = {
        "Evaluation Protocol": "Leave-One-Drop-Out (Novel Unseen Attack)",
        "ROC_AUC": f"{m_novel['ROC_AUC']:.4f}",
        "Detection Rate": f"{m_novel['Recall']:.1%}",
        "Over-refusal": f"{m_novel['Over_refusal_Rate']:.1%}"
    }

    t2_df = pd.DataFrame(continual_data + [novel_data])
    evaluator.print_markdown_table(f"TABLE 2: CONTINUAL ROUNDS & NOVEL ATTACK GENERALIZATION{eval_title_suffix}", t2_df)
    evaluator.save_and_sync_table("table2_continual_and_novel.csv", t2_df)
    if eval_split == "val":
        evaluator.save_and_sync_table("table2_val_continual_and_novel.csv", t2_df)

    # TABLE 3: Target Model (Qwen/Qwen3.5-4B) End-to-End Defense
    defender = TargetModelDefender(config)
    e2e_results = defender.evaluate_e2e_benchmark(
        test_df,
        p_cascade,
        p_vlm_clean,
        calibration_labels=y_calibration,
        calibration_hard_benign_mask=hard_benign_calibration,
        calibration_guard_probs=p_cascade_calibration,
        calibration_qlora_probs=p_vlm_clean_calibration,
    )
    evaluator.report_table3_end_to_end_target(e2e_results)
    if eval_split == "val":
        evaluator.save_and_sync_table("table3_val_end_to_end_4b.csv", pd.DataFrame(e2e_results))

    # ROC Curves Plot
    preds_dict = {
        "Reflexivity Heuristic": p_reflexivity,
        "Qwen3.5-2B Zero-shot": p_vlm_zero,
        "Qwen3.5-2B QLoRA Dirty": p_vlm_dirty,
        "Qwen3.5-2B QLoRA Clean": p_vlm_clean,
        "Cascade & Ensemble": p_cascade
    }
    evaluator.plot_roc_curves(y_test_true, preds_dict, "roc_curves.png")
    if eval_split == "val":
        evaluator.plot_roc_curves(y_test_true, preds_dict, "roc_curves_val.png")

    logger.info("====================================================================")
    logger.info(f" BENCHMARK COMPLETED SUCCESSFULLY! (Split: {eval_split.upper()})")
    logger.info(f" All result tables and ROC curves are saved to: {RESULTS_DIR}")
    if config.drive_backup_dir:
        logger.info(f" Results mirrored to Google Drive: {config.drive_backup_dir}")
    logger.info("====================================================================")

    # =========================================================================
    # STAGE 6: Competition Submission Generation on test.csv
    # =========================================================================
    if args.full_run or args.predict_test or (args.test_file is not None):
        generate_competition_submission(
            config=config,
            reflex_clf=reflex_clf,
            vlm_branch=vlm_branch,
            ensemble=ensemble,
            cascade=cascade,
            opt_thresh=opt_thresh_best,
            evaluator=evaluator,
            custom_test_file=args.test_file,
            custom_sub_file=args.sample_submission
        )

    # Auto-synchronize LaTeX Presentation slides with generated CSV results
    try:
        from presentation.update_slides import update_presentation_slides
        update_presentation_slides(use_val=(eval_split == "val"))
    except Exception as e:
        logger.debug(f"Could not auto-sync presentation slides: {e}")


def generate_competition_submission(
    config,
    reflex_clf: ReflexivityClassifier,
    vlm_branch: QwenVLMBranch,
    ensemble: MultiSignalRankEnsemble,
    cascade: CascadeGatekeeper,
    opt_thresh: float,
    evaluator: BenchmarkEvaluator,
    custom_test_file: Optional[str] = None,
    custom_sub_file: Optional[str] = None
):
    """
    Executes end-to-end inference on the official competition test.csv,
    generates submission.csv matching sample_submission.csv schema,
    saves to results/ and mirrors to Google Drive.
    """
    logger.info("")
    logger.info("====================================================================")
    logger.info(" >>> [STAGE 6] Generating Competition Submission on test.csv...")
    logger.info("====================================================================")

    test_df, sample_sub, test_path, sub_path = load_rmit_hackathon_2025_test_dataset(
        custom_test_path=custom_test_file,
        custom_sub_path=custom_sub_file
    )

    if test_df is None or len(test_df) == 0:
        logger.warning(
            "[Submission] test.csv not found in candidate paths. "
            "Please ensure test.csv is placed in data/downloads/rmit-hackathon-2025/ "
            "or provide --test_file <path_to_test.csv>."
        )
        return

    logger.info(f"[Submission] Loaded test dataset: {len(test_df)} samples from {test_path}")
    if sample_sub is not None:
        logger.info(f"[Submission] Loaded reference sample_submission: {len(sample_sub)} samples from {sub_path}")
        logger.info(f"[Submission] Submission columns format: {list(sample_sub.columns)}")

    # 1. Tier 1 Reflexivity Classifier
    logger.info("[Submission] 1/5 Computing Fast Reflexivity Probabilities...")
    p_reflex = reflex_clf.predict_proba(test_df["prompt"].tolist())

    # 2. Tier 2 Qwen3.5-2B QLoRA Guard
    logger.info("[Submission] 2/5 Computing Qwen3.5-2B QLoRA Logits...")
    p_qlora, vlm_latency = vlm_branch.predict_logits_single_step(test_df)

    # 3. Zero-shot Signal
    logger.info("[Submission] 3/5 Computing Zero-shot Anchor Signal...")
    if hasattr(vlm_branch.model, "disable_adapter"):
        try:
            with vlm_branch.model.disable_adapter():
                p_zero, _ = vlm_branch.predict_logits_single_step(test_df)
        except Exception:
            p_zero = p_qlora
    else:
        p_zero = p_qlora

    # 4. OOD Anomaly Scores
    logger.info("[Submission] 4/5 Computing OOD Anomaly Scores...")
    p_ood = vlm_branch.compute_ood_scores(test_df)

    # 5. Multi-Signal Ensemble & Cascade
    logger.info("[Submission] 5/5 Applying Optimized Ensemble & Cascade Gatekeeper...")
    p_ensemble = ensemble.predict(p_qlora, p_zero, p_ood, p_reflex)
    p_cascade, stages, avg_lat, tier2_rate = cascade.evaluate_cascade(
        test_df, p_reflex, p_ensemble, reflex_latency_ms=0.5, deep_latency_ms=vlm_latency
    )
    tier1_rate = 1.0 - tier2_rate

    # 6. Build Submission DataFrame matching sample_submission.csv
    target_col = "TARGET"
    id_col = "Id"
    if sample_sub is not None and len(sample_sub.columns) >= 2:
        id_col = sample_sub.columns[0]
        target_col = sample_sub.columns[1]

    # Map predictions by Id to preserve exact row ordering from sample_submission.csv
    pred_map = dict(zip(test_df["Id"], np.round(p_cascade, 6)))
    if sample_sub is not None:
        submission_df = pd.DataFrame()
        submission_df[id_col] = sample_sub[id_col]
        submission_df[target_col] = sample_sub[id_col].map(pred_map).fillna(0.5)
    else:
        submission_df = pd.DataFrame({
            id_col: test_df["Id"],
            target_col: np.round(p_cascade, 6)
        })

    # Validate output
    expected_rows = len(sample_sub) if sample_sub is not None else len(test_df)
    assert len(submission_df) == expected_rows, f"Row count mismatch: {len(submission_df)} vs {expected_rows}"
    assert not submission_df.isna().any().any(), "Submission DataFrame contains NaN values!"
    if sample_sub is not None:
        unmatched_ids = submission_df[target_col].eq(0.5) & ~sample_sub[id_col].isin(test_df["Id"])
        if unmatched_ids.any():
            raise ValueError("sample_submission.csv contains Id values absent from test.csv.")

    # 7. Save Submission Files
    sub_primary = RESULTS_DIR / "submission.csv"
    sub_probs = RESULTS_DIR / "submission_probabilities.csv"
    sub_binary = RESULTS_DIR / "submission_binary.csv"

    submission_df.to_csv(sub_primary, index=False)
    submission_df.to_csv(sub_probs, index=False)

    sub_binary_df = submission_df.copy()
    sub_binary_df[target_col] = (submission_df[target_col] >= opt_thresh).astype(int)
    sub_binary_df.to_csv(sub_binary, index=False)

    evaluator.save_and_sync_table("submission.csv", submission_df)
    evaluator.save_and_sync_table("submission_probabilities.csv", submission_df)
    evaluator.save_and_sync_table("submission_binary.csv", sub_binary_df)

    # 8. Print Official Submission Report
    banner = "=" * 80
    logger.info(f"\n{banner}\n COMPETITION SUBMISSION GENERATED SUCCESSFULLY\n{banner}")
    logger.info(f" Primary Submission File : {sub_primary}")
    logger.info(f" Binary Sub File         : {sub_binary} (Optimal Threshold: {opt_thresh:.4f})")
    if config.drive_backup_dir:
        logger.info(f" Mirrored to Drive       : {config.drive_backup_dir}/submission.csv")
    logger.info(f" Total Rows              : {len(submission_df)}")
    logger.info(f" Columns Format          : {list(submission_df.columns)}")
    logger.info(
        f" Probability Stats       : Min={submission_df[target_col].min():.4f} | "
        f"Mean={submission_df[target_col].mean():.4f} | "
        f"Median={submission_df[target_col].median():.4f} | "
        f"Max={submission_df[target_col].max():.4f}"
    )
    logger.info(f" Tier 1 Fast Screened    : {tier1_rate:.1%}")
    logger.info(f" Tier 2 Deep VLM         : {tier2_rate:.1%}")
    logger.info(f" Avg Test Latency        : {avg_lat:.2f} ms/query")
    logger.info("-" * 80)
    logger.info(" Preview First 10 Rows (submission.csv):")
    logger.info("\n" + submission_df.head(10).to_string(index=False))
    logger.info(f"{banner}\n")


if __name__ == "__main__":
    main()

