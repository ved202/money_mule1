"""
src/evaluation/evaluator.py
============================
Computes and reports all model performance metrics.

Metrics reported
----------------
  Per-model   : Precision, Recall, F1, PR-AUC, ROC-AUC
  System-level: Early Detection Gain, Community Detection Recall
  Output      : outputs/reports/evaluation_report.txt
"""

import pandas as pd
import numpy as np
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    average_precision_score, roc_auc_score,
    classification_report, confusion_matrix
)
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from config.config import *


def evaluate_model(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    name:   str,
    verbose: bool = True,
) -> dict:
    """Compute and return full metric set for one model."""
    p   = precision_score(y_true, y_pred, zero_division=0)
    r   = recall_score(y_true, y_pred, zero_division=0)
    f1  = f1_score(y_true, y_pred, zero_division=0)
    pra = average_precision_score(y_true, y_prob) if y_true.sum() > 0 else 0.0
    roc = roc_auc_score(y_true, y_prob)          if y_true.sum() > 0 else 0.0

    cm  = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn  = cm[0, 0]; fp = cm[0, 1]; fn = cm[1, 0]; tp = cm[1, 1]

    if verbose:
        print(f"\n{'â”€'*50}")
        print(f"  {name}")
        print(f"{'â”€'*50}")
        print(classification_report(y_true, y_pred,
                                    target_names=["Normal", "Mule"], digits=4))
        print(f"  PR-AUC  : {pra:.4f}")
        print(f"  ROC-AUC : {roc:.4f}")
        print(f"  TP={tp}  FP={fp}  FN={fn}  TN={tn}")

    return {
        "Model":     name,
        "Precision": round(p,   4),
        "Recall":    round(r,   4),
        "F1":        round(f1,  4),
        "PR-AUC":    round(pra, 4),
        "ROC-AUC":   round(roc, 4),
        "TP": tp, "FP": fp, "FN": fn, "TN": tn,
    }


def evaluate_early_detection(lifecycle_df: pd.DataFrame) -> dict:
    """Compute Early Detection Gain â€” mules caught before Laundering."""
    if "is_fraud" not in lifecycle_df.columns:
        return {}

    mules = lifecycle_df[lifecycle_df["is_fraud"] == 1]
    total = len(mules)
    early = int(mules["early_flag"].sum()) if "early_flag" in mules.columns else 0
    edg   = early / max(total, 1)

    stage_dist = mules["lifecycle_stage"].value_counts().to_dict()

    print(f"\n{'â”€'*50}")
    print(f"  Early Detection Gain")
    print(f"{'â”€'*50}")
    print(f"  Total mule accounts   : {total:,}")
    print(f"  Caught early          : {early:,} ({edg:.2%})")
    print(f"  Stage distribution    : {stage_dist}")
    print("  Status                : heuristic / not independently validated")

    return {
        "total_mules": total,
        "early_detected": early,
        "early_detection_gain": round(edg, 4),
        "stage_distribution": stage_dist,
        "status": "heuristic",
    }


def evaluate_community_detection(
    partition:      dict,
    feature_matrix: pd.DataFrame,
    comm_df:        pd.DataFrame,
) -> dict:
    """
    Evaluate community detection quality:
      - What fraction of mule accounts are in suspicious communities?
      - What fraction of suspicious communities contain mules?
    """
    if "is_fraud" not in feature_matrix.columns:
        return {}

    fm = feature_matrix.set_index("account") if "account" in feature_matrix.columns \
         else feature_matrix

    suspicious_cids = set(
        comm_df[comm_df["is_suspicious"] == 1]["community_id"].tolist()
    )

    mule_accounts   = set(fm[fm["is_fraud"] == 1].index)
    total_mules     = len(mule_accounts)

    mules_in_susp   = sum(
        1 for acc in mule_accounts
        if partition.get(acc, -1) in suspicious_cids
    )
    community_recall = mules_in_susp / max(total_mules, 1)

    print(f"\n{'â”€'*50}")
    print(f"  Community Detection")
    print(f"{'â”€'*50}")
    print(f"  Suspicious communities: {len(suspicious_cids):,}")
    print(f"  Mules in susp. comms  : {mules_in_susp}/{total_mules} "
          f"({community_recall:.2%})")

    return {
        "n_suspicious_communities": len(suspicious_cids),
        "mules_in_suspicious":      mules_in_susp,
        "community_recall":         round(community_recall, 4),
    }


def compile_report(
    model_metrics:   list[dict],
    early_detection: dict,
    community_eval:  dict,
    suspicious_accounts: list,
) -> str:
    """Build and save the full evaluation report."""
    lines = []
    lines.append("=" * 70)
    lines.append("  MULE DETECTION SYSTEM â€” EVALUATION REPORT")
    lines.append("=" * 70)

    lines.append("\nâ”€â”€ MODEL PERFORMANCE (test set) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€")
    df = pd.DataFrame(model_metrics)
    lines.append(df[["Model","Precision","Recall","F1","PR-AUC","ROC-AUC"]]
                 .to_string(index=False))

    perf_df = df[df["Model"].isin(["RandomForest", "GradientBoosting", "GNN-SAGE", "GNN-GAT"])].copy()
    if not perf_df.empty and "PR-AUC" in perf_df.columns:
        best_idx = perf_df["PR-AUC"].astype(float).idxmax()
        best = perf_df.loc[best_idx]
        lines.append("\nâ”€â”€ RECOMMENDED MODEL â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€")
        lines.append(
            f"  Best validation model : {best['Model']} "
            f"(PR-AUC={float(best['PR-AUC']):.4f}, ROC-AUC={float(best['ROC-AUC']):.4f})"
        )
        lines.append("  Recommendation        : Use this as the primary fraud detector on PaySim.")

    if early_detection:
        lines.append("\nâ”€â”€ EARLY DETECTION (Objective 2) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€")
        lines.append(f"  Total mule accounts      : {early_detection.get('total_mules', 0):,}")
        lines.append(f"  Caught at early stage    : {early_detection.get('early_detected', 0):,}")
        lines.append(f"  Early Detection Gain     : {early_detection.get('early_detection_gain', 0):.2%}")
        lines.append(f"  Stage distribution       : {early_detection.get('stage_distribution', {})}")
        lines.append(f"  Status                   : {early_detection.get('status', 'unvalidated')}")
        lines.append("  Interpretation           : Treat lifecycle output as heuristic until separately validated.")

    if community_eval:
        lines.append("\nâ”€â”€ COMMUNITY DETECTION â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€")
        lines.append(f"  Suspicious communities   : {community_eval.get('n_suspicious_communities', 0):,}")
        lines.append(f"  Community recall         : {community_eval.get('community_recall', 0):.2%}")

    lines.append(f"\nâ”€â”€ SUSPICIOUS ACCOUNT SUMMARY â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€")
    lines.append(f"  Total suspicious accounts: {len(suspicious_accounts):,}")

    report = "\n".join(lines)
    path   = OUTPUTS_REPORTS / "evaluation_report.txt"
    with open(path, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"\n  Full report saved â†’ {path}")
    print(report)
    return report
