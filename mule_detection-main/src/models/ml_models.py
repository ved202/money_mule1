"""
src/models/ml_models.py
========================
Trains three complementary ML models for mule account detection.

Model roles
-----------
  IsolationForest  : Unsupervised anomaly detection.
                     Flags statistically unusual accounts without labels.
                     Use when labels are sparse or unavailable.

  RandomForest     : Supervised ensemble classifier.
                     High interpretability (feature importances).
                     Robust to class imbalance via class_weight='balanced'.

  GradientBoosting : Supervised sequential ensemble.
                     Typically highest precision on structured tabular data.
                     More sensitive to hyperparameter tuning than RF.

All three are combined into an ensemble via soft-voting on probability scores.
"""

import pandas as pd
import numpy as np
import pickle
import warnings
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    IsolationForest,
    VotingClassifier,
)
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import (
    train_test_split, StratifiedKFold, cross_val_score
)
from sklearn.metrics import (
    classification_report, precision_recall_curve,
    average_precision_score, roc_auc_score,
    confusion_matrix, f1_score
)
from sklearn.calibration import CalibratedClassifierCV
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from config.config import *

warnings.filterwarnings("ignore")


# ── Feature column resolver ───────────────────────────────────────────────────
def _get_feature_cols(df: pd.DataFrame) -> list[str]:
    drop = {"account", "is_fraud", "source", "tx_id"}
    return [c for c in df.columns
            if c not in drop and pd.api.types.is_numeric_dtype(df[c])]


def _time_aware_train_test_split(
    feature_matrix: pd.DataFrame,
    feat_cols: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Prefer a chronological split when temporal metadata is available."""
    if "last_seen_ts" in feature_matrix.columns:
        ordered = feature_matrix.sort_values("last_seen_ts").reset_index(drop=True)
        split_idx = int(len(ordered) * (1 - TEST_SIZE))
        split_idx = min(max(split_idx, 1), len(ordered) - 1)
        train_df = ordered.iloc[:split_idx]
        test_df = ordered.iloc[split_idx:]

        if train_df["is_fraud"].nunique() > 1 and test_df["is_fraud"].nunique() > 1:
            return (
                train_df[feat_cols].values,
                test_df[feat_cols].values,
                train_df["is_fraud"].values,
                test_df["is_fraud"].values,
                ordered["account"].values,
            )

    X = feature_matrix[feat_cols].values
    y = feature_matrix["is_fraud"].values
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=TEST_SIZE,
        random_state=RANDOM_STATE, stratify=y
    )
    return X_tr, X_te, y_tr, y_te, feature_matrix["account"].values


def _fit_threshold(
    model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    sample_weight: np.ndarray | None = None,
) -> float:
    """
    Fit on a train subset, tune decision threshold on a validation subset,
    then refit on the full training fold.
    """
    if len(np.unique(y_train)) < 2 or len(y_train) < 10:
        if sample_weight is None:
            model.fit(X_train, y_train)
        else:
            model.fit(X_train, y_train, sample_weight=sample_weight)
        return 0.5

    X_fit, X_val, y_fit, y_val, sw_fit, _ = train_test_split(
        X_train,
        y_train,
        sample_weight if sample_weight is not None else np.ones(len(y_train)),
        test_size=VALIDATION_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_train,
    )

    if sample_weight is None:
        model.fit(X_fit, y_fit)
        fit_kwargs = {}
    else:
        model.fit(X_fit, y_fit, sample_weight=sw_fit)
        fit_kwargs = {"sample_weight": sample_weight}

    val_prob = model.predict_proba(X_val)[:, 1]
    precision, recall, thresholds = precision_recall_curve(y_val, val_prob)
    if len(thresholds) == 0:
        threshold = 0.5
    else:
        f1_scores = (2 * precision[:-1] * recall[:-1]) / (
            precision[:-1] + recall[:-1] + 1e-9
        )
        threshold = float(thresholds[int(np.nanargmax(f1_scores))])

    model.fit(X_train, y_train, **fit_kwargs)
    return threshold


# ═══════════════════════════════════════════════════════════════════════════════
# 1. ISOLATION FOREST (unsupervised)
# ═══════════════════════════════════════════════════════════════════════════════

def train_isolation_forest(
    feature_matrix: pd.DataFrame,
    contamination: float = None,
) -> tuple:
    """
    Train Isolation Forest on all accounts (no labels required).

    Returns (model, scaler, anomaly_scores_series, predictions_series)
    """
    print("\n--- Training Isolation Forest ---")
    feat_cols = _get_feature_cols(feature_matrix)
    X = feature_matrix[feat_cols].values

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Use known fraud rate as contamination if available
    if contamination is None:
        if "is_fraud" in feature_matrix.columns and feature_matrix["is_fraud"].sum() > 0:
            contamination = float(np.clip(feature_matrix["is_fraud"].mean(), 0.001, 0.1))
        else:
            contamination = "auto"

    iso = IsolationForest(
        n_estimators=ISO_N_ESTIMATORS,
        contamination=contamination,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    iso.fit(X_scaled)

    raw_scores  = iso.score_samples(X_scaled)   # lower = more anomalous
    anom_scores = -raw_scores                    # flip: higher = more suspicious
    iso_preds   = (iso.predict(X_scaled) == -1).astype(int)

    accs = feature_matrix["account"].values if "account" in feature_matrix.columns \
           else feature_matrix.index

    scores_s = pd.Series(anom_scores, index=accs, name="iso_score")
    preds_s  = pd.Series(iso_preds,   index=accs, name="iso_pred")

    if "is_fraud" in feature_matrix.columns:
        y_true = feature_matrix["is_fraud"].values
        if np.unique(y_true).size > 1 and y_true.sum() > 0:
            pr_auc = average_precision_score(y_true, anom_scores)
            roc    = roc_auc_score(y_true, anom_scores)
            print(f"  PR-AUC  : {pr_auc:.4f}")
            print(f"  ROC-AUC : {roc:.4f}")
            print(classification_report(y_true, iso_preds,
                                        target_names=["Normal", "Mule"], digits=4))

    print(f"  Flagged anomalies: {iso_preds.sum():,} / {len(iso_preds):,}")
    return iso, scaler, scores_s, preds_s


# ═══════════════════════════════════════════════════════════════════════════════
# 2. RANDOM FOREST (supervised)
# ═══════════════════════════════════════════════════════════════════════════════

def train_random_forest(
    feature_matrix: pd.DataFrame,
) -> tuple:
    """
    Train Random Forest classifier. Returns (model, scaler, results_dict).
    """
    print("\n--- Training Random Forest ---")
    feat_cols = _get_feature_cols(feature_matrix)
    X = feature_matrix[feat_cols].values

    X_tr, X_te, y_tr, y_te, accs = _time_aware_train_test_split(
        feature_matrix,
        feat_cols,
    )

    rf = RandomForestClassifier(
        n_estimators=RF_N_ESTIMATORS,
        max_depth=RF_MAX_DEPTH,
        min_samples_leaf=RF_MIN_SAMPLES_LEAF,
        class_weight=RF_CLASS_WEIGHT,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    threshold = _fit_threshold(rf, X_tr, y_tr)

    y_prob = rf.predict_proba(X_te)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    pr_auc  = average_precision_score(y_te, y_prob)
    roc_auc = roc_auc_score(y_te, y_prob)

    print(f"  PR-AUC  : {pr_auc:.4f}")
    print(f"  ROC-AUC : {roc_auc:.4f}")
    print(classification_report(y_te, y_pred,
                                target_names=["Normal", "Mule"], digits=4))

    # Feature importances
    fi = pd.Series(rf.feature_importances_, index=feat_cols)
    fi = fi.sort_values(ascending=False)
    print(f"\n  Top 10 features:")
    print(fi.head(10).to_string())

    # Full-dataset probabilities
    all_proba = rf.predict_proba(X)[:, 1]
    all_preds = (all_proba >= threshold).astype(int)
    accs      = feature_matrix["account"].values

    results = {
        "pr_auc":     pr_auc,
        "roc_auc":    roc_auc,
        "proba":      pd.Series(all_proba, index=accs, name="rf_proba"),
        "preds":      pd.Series(all_preds, index=accs, name="rf_pred"),
        "feat_importance": fi,
        "threshold":  threshold,
        "y_test":     y_te,
        "y_prob_test": y_prob,
        "y_pred_test": y_pred,
    }
    return rf, None, results


# ═══════════════════════════════════════════════════════════════════════════════
# 3. GRADIENT BOOSTING (supervised)
# ═══════════════════════════════════════════════════════════════════════════════

def train_gradient_boosting(
    feature_matrix: pd.DataFrame,
) -> tuple:
    """
    Train Gradient Boosting classifier. Returns (model, scaler, results_dict).
    """
    print("\n--- Training Gradient Boosting ---")
    feat_cols = _get_feature_cols(feature_matrix)
    X = feature_matrix[feat_cols].values
    y = feature_matrix["is_fraud"].values

    X_tr, X_te, y_tr, y_te, _ = _time_aware_train_test_split(
        feature_matrix,
        feat_cols,
    )

    # Class weight via sample_weight
    n_neg = (y_tr == 0).sum()
    n_pos = (y_tr == 1).sum()
    sample_weight = np.where(y_tr == 1, n_neg / max(n_pos, 1), 1.0)

    gb = GradientBoostingClassifier(
        n_estimators=GB_N_ESTIMATORS,
        learning_rate=GB_LEARNING_RATE,
        max_depth=GB_MAX_DEPTH,
        random_state=RANDOM_STATE,
    )
    threshold = _fit_threshold(gb, X_tr, y_tr, sample_weight=sample_weight)

    y_prob = gb.predict_proba(X_te)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    pr_auc  = average_precision_score(y_te, y_prob)
    roc_auc = roc_auc_score(y_te, y_prob)

    print(f"  PR-AUC  : {pr_auc:.4f}")
    print(f"  ROC-AUC : {roc_auc:.4f}")
    print(classification_report(y_te, y_pred,
                                target_names=["Normal", "Mule"], digits=4))

    all_proba = gb.predict_proba(X)[:, 1]
    all_preds = (all_proba >= threshold).astype(int)
    accs      = feature_matrix["account"].values

    results = {
        "pr_auc":      pr_auc,
        "roc_auc":     roc_auc,
        "proba":       pd.Series(all_proba, index=accs, name="gb_proba"),
        "preds":       pd.Series(all_preds, index=accs, name="gb_pred"),
        "threshold":   threshold,
        "y_test":      y_te,
        "y_prob_test": y_prob,
        "y_pred_test": y_pred,
    }
    return gb, None, results


# ═══════════════════════════════════════════════════════════════════════════════
# 4. ENSEMBLE (soft voting)
# ═══════════════════════════════════════════════════════════════════════════════

def build_ensemble_scores(
    iso_scores: pd.Series,
    rf_proba:   pd.Series,
    gb_proba:   pd.Series,
    gnn_proba:  pd.Series | None = None,
    weights:    tuple | None = None,
) -> pd.Series:
    """
    Combine model scores into a single ensemble suspicion score.
    Normalise each model's scores to [0, 1] before weighting.
    """
    def _norm(s):
        mn, mx = s.min(), s.max()
        return (s - mn) / (mx - mn + 1e-9)

    common = iso_scores.index.intersection(rf_proba.index).intersection(gb_proba.index)

    components = [
        _norm(iso_scores.loc[common]).rename("iso"),
        rf_proba.loc[common].rename("rf"),
        gb_proba.loc[common].rename("gb"),
    ]
    component_weights = list(weights or (0.20, 0.45, 0.35))

    if gnn_proba is not None:
        common = common.intersection(gnn_proba.index)
        components = [
            _norm(iso_scores.loc[common]).rename("iso"),
            rf_proba.loc[common].rename("rf"),
            gb_proba.loc[common].rename("gb"),
            gnn_proba.loc[common].rename("gnn"),
        ]
        component_weights = list(weights or (0.15, 0.35, 0.25, 0.25))

    score = pd.Series(0.0, index=common, dtype=float)
    total_weight = sum(component_weights)
    for component, weight in zip(components, component_weights):
        score = score.add(component * weight, fill_value=0)

    return (score / total_weight).rename("ensemble_score")


# ═══════════════════════════════════════════════════════════════════════════════
# 5. SAVE / LOAD
# ═══════════════════════════════════════════════════════════════════════════════

def save_model(obj, name: str):
    path = OUTPUTS_MODELS / f"{name}.pkl"
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    print(f"  Saved model → {path}")


def load_model(name: str):
    path = OUTPUTS_MODELS / f"{name}.pkl"
    with open(path, "rb") as f:
        return pickle.load(f)
