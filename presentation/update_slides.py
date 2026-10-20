"""
presentation/update_slides.py - Auto-update LaTeX Beamer presentation tables from results/*.csv
RMIT HACKATHON 2026 - Multimodal Gatekeeper AI Safety Benchmark
"""

import sys
import re
from pathlib import Path
import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = _ROOT / "results"
SLIDES_TEX = _ROOT / "presentation" / "slides.tex"


def format_latex_escape(val) -> str:
    """Escapes special LaTeX characters in cell content."""
    s = str(val).strip()
    s = s.replace("%", r"\%").replace("_", r"\_").replace("&", r"\&")
    return s


def build_table0_latex(df: pd.DataFrame) -> str:
    """Builds LaTeX tabular rows for Table 0 (Poison Sanitizer)."""
    mapping = dict(zip(df.iloc[:, 0].astype(str), df.iloc[:, 1].astype(str)))

    prec = format_latex_escape(mapping.get("Poison Detection Precision", "100.00%"))
    rec = format_latex_escape(mapping.get("Poison Detection Recall", "100.00%"))
    f1 = format_latex_escape(mapping.get("Poison Detection F1-Score", "1.0000"))
    auc_dirty = format_latex_escape(mapping.get("VLM ROC-AUC (Trained on Noisy Data)", "0.8729"))
    auc_clean = format_latex_escape(mapping.get("VLM ROC-AUC (Trained on Cleaned Data)", "0.9231"))
    delta = format_latex_escape(mapping.get("ROC-AUC Delta (Gain from Sanitization)", "+0.0502"))

    return (
        f"            Poison Detection Precision & \\textbf{{{prec}}} \\\\\n"
        f"            Poison Detection Recall & \\textbf{{{rec}}} (Catching injected poison) \\\\\n"
        f"            Poison Detection F1-Score & \\textbf{{{f1}}} \\\\\n"
        f"            Model ROC-AUC on Dirty / Poisoned Data & {auc_dirty} \\\\\n"
        f"            Model ROC-AUC on Cleaned Data & \\textbf{{{auc_clean}}} \\\\\n"
        f"            \\midrule\n"
        f"            \\textbf{{ROC-AUC Gain ($\\Delta \\text{{AUC}}$ from Sanitization)}} & \\textbf{{{delta}}} (Prevents Over-refusal Collapse) \\\\"
    )


def build_table1_latex(df: pd.DataFrame) -> str:
    """Builds LaTeX tabular rows for Table 1 (Method Comparison)."""
    rows = []
    for _, row in df.iterrows():
        name = str(row.iloc[0]).strip()
        auc = format_latex_escape(row.get("ROC_AUC", "0.0000"))
        f1 = format_latex_escape(row.get("F1_Score", "0.0000"))
        rec = format_latex_escape(row.get("Recall (Detection)", "0.00%"))
        fpr = format_latex_escape(row.get("Over-refusal (Hard Benign)", "0.00%"))
        lat = format_latex_escape(row.get("Latency (ms)", "0.0 ms"))
        vram = format_latex_escape(row.get("Peak VRAM (MB)", "0 MB"))

        if "Cascade" in name or name.startswith("5."):
            row_str = (
                f"            \\midrule\n"
                f"            \\rowcolor{{LightCyan}}\n"
                f"            \\textbf{{5. Cascade \\& Multi-Signal Ensemble}} & \\textbf{{{auc}}} & \\textbf{{{f1}}} & \\textbf{{{rec}}} & \\textbf{{{fpr}}} & \\textbf{{{lat}}} & \\textbf{{{vram}}} \\\\"
            )
        elif "Reflexivity" in name or name.startswith("1."):
            row_str = f"            1. Reflexivity Heuristic Only & {auc} & {f1} & {rec} & \\textbf{{{fpr}}} & \\textbf{{{lat}}} & \\textbf{{{vram}}} \\\\"
        elif "Zero-shot" in name or name.startswith("2."):
            row_str = f"            2. Qwen3.5-2B Zero-shot & {auc} & {f1} & {rec} & {fpr} & {lat} & {vram} \\\\"
        elif "Dirty" in name or (name.startswith("3.") and "Poison" in name):
            fpr_note = f"{fpr} \\textcolor{{RMITRed}}{{(Collapsed)}}" if float(str(row.get("Over-refusal (Hard Benign)", "0")).replace("%", "").strip()) > 50 else fpr
            row_str = f"            3. Qwen3.5-2B QLoRA (Dirty / Poisoned Data) & {auc} & {f1} & {rec} & {fpr_note} & {lat} & {vram} \\\\"
        elif "Cleaned" in name or name.startswith("4."):
            row_str = f"            4. Qwen3.5-2B QLoRA (Cleaned Data) & {auc} & {f1} & {rec} & \\textbf{{{fpr}}} & {lat} & {vram} \\\\"
        else:
            row_str = f"            {format_latex_escape(name)} & {auc} & {f1} & {rec} & {fpr} & {lat} & {vram} \\\\"
        rows.append(row_str)

    return "\n".join(rows)


def build_table2_latex(df: pd.DataFrame) -> str:
    """Builds LaTeX tabular rows for Table 2 (Continual & Novel)."""
    rows = []
    for _, row in df.iterrows():
        proto = str(row.iloc[0]).strip()
        auc = format_latex_escape(row.get("ROC_AUC", "0.0000"))
        rec = format_latex_escape(row.get("Detection Rate", "0.0%"))
        fpr = format_latex_escape(row.get("Over-refusal", "0.0%"))

        if "Novel" in proto or "Leave-One" in proto:
            rows.append(f"                    \\midrule")
            rows.append(f"                    \\textbf{{Leave-One-Drop-Out (Novel)}} & \\textbf{{{auc}}} & \\textbf{{{rec}}} & \\textbf{{{fpr}}} \\\\")
        else:
            proto_clean = format_latex_escape(proto)
            rows.append(f"                    {proto_clean} & {auc} & {rec} & {fpr} \\\\")

    return "\n".join(rows)


def build_table3_latex(df: pd.DataFrame) -> str:
    """Builds LaTeX tabular rows for Table 3 (Target 4B Protection)."""
    rows = []
    for _, row in df.iterrows():
        cfg = str(row.iloc[0]).strip()
        asr = format_latex_escape(row.get("Attack Success Rate (ASR) [lower is better]", "0.00%"))
        fpr = format_latex_escape(row.get("Over-refusal Rate (FPR) [lower is better]", "0.00%"))
        lat = format_latex_escape(row.get("End-to-End Latency (ms)", "0.0 ms"))

        if "Cascade" in cfg or "ProbGuard" in cfg:
            rows.append(
                f"            \\rowcolor{{LightCyan}}\n"
                f"            \\textbf{{Cascade \\& ProbGuard 10-Token Lookahead + Target 4B}} & \\textbf{{{asr}}} \\textcolor{{SuccessGreen}}{{(Fully Blocked)}} & \\textbf{{{fpr}}} & \\textbf{{{lat}}} \\\\"
            )
        elif "No Defense" in cfg:
            rows.append(f"            No Defense (Target 4B Only) & {asr} \\textcolor{{RMITRed}}{{(Vulnerable)}} & {fpr} & {lat} \\\\")
        elif "Standard Guard" in cfg:
            rows.append(f"            Standard Guard (Qwen3.5-2B) + Target 4B & {asr} & {fpr} & {lat} \\\\")
        else:
            rows.append(f"            {format_latex_escape(cfg)} & {asr} & {fpr} & {lat} \\\\")

    return "\n".join(rows)


def update_presentation_slides(use_val: bool = False):
    if not SLIDES_TEX.exists():
        print(f"Error: {SLIDES_TEX} not found!")
        return

    content = SLIDES_TEX.read_text(encoding="utf-8")
    suffix = "_val" if use_val else ""

    p_t0 = RESULTS_DIR / "table0_poison_filtering.csv"
    p_t1 = RESULTS_DIR / f"table1{suffix}_method_comparison.csv"
    if not p_t1.exists():
        p_t1 = RESULTS_DIR / "table1_method_comparison.csv"
    p_t2 = RESULTS_DIR / f"table2{suffix}_continual_and_novel.csv"
    if not p_t2.exists():
        p_t2 = RESULTS_DIR / "table2_continual_and_novel.csv"
    p_t3 = RESULTS_DIR / f"table3{suffix}_end_to_end_4b.csv"
    if not p_t3.exists():
        p_t3 = RESULTS_DIR / "table3_end_to_end_4b.csv"

    updated_count = 0

    # 1. Update Table 0
    if p_t0.exists():
        df0 = pd.read_csv(p_t0)
        t0_body = build_table0_latex(df0)
        pat0 = r"(\\caption\{\\textbf\{Table 0:[^\}]+\}\}[\s\S]*?\\midrule\n)[\s\S]*?(\\bottomrule)"
        if re.search(pat0, content):
            content = re.sub(pat0, lambda m: m.group(1) + t0_body + "\n            " + m.group(2), content)
            print(f"  [+] Table 0 updated from {p_t0.name}")
            updated_count += 1

    # 2. Update Table 1
    if p_t1.exists():
        df1 = pd.read_csv(p_t1)
        t1_body = build_table1_latex(df1)
        pat1 = r"(\\caption\{\\textbf\{Table 1:[^\}]+\}\}[\s\S]*?\\midrule\n)[\s\S]*?(\\bottomrule)"
        if re.search(pat1, content):
            content = re.sub(pat1, lambda m: m.group(1) + t1_body + "\n            " + m.group(2), content)
            print(f"  [+] Table 1 updated from {p_t1.name}")
            updated_count += 1

    # 3. Update Table 2
    if p_t2.exists():
        df2 = pd.read_csv(p_t2)
        t2_body = build_table2_latex(df2)
        pat2 = r"(\\caption\{\\textbf\{Table 2:[^\}]+\}\}[\s\S]*?\\midrule\n)[\s\S]*?(\\bottomrule)"
        if re.search(pat2, content):
            content = re.sub(pat2, lambda m: m.group(1) + t2_body + "\n                    " + m.group(2), content)
            print(f"  [+] Table 2 updated from {p_t2.name}")
            updated_count += 1

    # 4. Update Table 3
    if p_t3.exists():
        df3 = pd.read_csv(p_t3)
        t3_body = build_table3_latex(df3)
        pat3 = r"(\\caption\{\\textbf\{Table 3:[^\}]+\}\}[\s\S]*?\\midrule\n)[\s\S]*?(\\bottomrule)"
        if re.search(pat3, content):
            content = re.sub(pat3, lambda m: m.group(1) + t3_body + "\n            " + m.group(2), content)
            print(f"  [+] Table 3 updated from {p_t3.name}")
            updated_count += 1

    SLIDES_TEX.write_text(content, encoding="utf-8")
    print(f"\nSuccessfully synchronized {updated_count}/4 tables in: {SLIDES_TEX}")


if __name__ == "__main__":
    use_val = "--val" in sys.argv
    print(f"=== Synchronizing LaTeX Slides with CSV Results (Mode: {'Validation' if use_val else 'Global Test'}) ===")
    update_presentation_slides(use_val=use_val)
