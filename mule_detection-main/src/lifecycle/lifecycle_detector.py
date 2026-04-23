"""
src/lifecycle/lifecycle_detector.py   (v2 â€” improved rules + ML stage classifier)
====================================================================================
Objective 2: Early-Stage Mule Lifecycle Detection
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from config.config import *

LIFECYCLE_STAGES  = ["Dormant", "Recruitment", "Activation", "Laundering", "Exit", "Normal"]
STAGE_RISK        = {"Dormant":0,"Normal":0,"Recruitment":2,"Activation":3,"Laundering":5,"Exit":4}

_STAGE_FEATURES = [
    "n_sent","n_recv","tx_count","burst_ratio","weekly_cv",
    "max_gap_days","sudden_wakeup","small_round_txns",
    "passthrough_ratio","tx_velocity_1d","tx_velocity_7d",
    "tx_velocity_30d","active_days","max_weekly_txns","mean_weekly_txns",
    "fanout_ratio","degree_ratio",
]


def classify_lifecycle_stage(row: pd.Series) -> str:
    n_tx   = float(row.get("tx_count",          0))
    n_sent = float(row.get("n_sent",            0))
    n_recv = float(row.get("n_recv",            0))
    gap    = float(row.get("max_gap_days",       0))
    burst  = float(row.get("burst_ratio",        1))
    cv     = float(row.get("weekly_cv",          0))
    pass_r = float(row.get("passthrough_ratio",  0))
    srt    = float(row.get("small_round_txns",   0))
    sw     = float(row.get("sudden_wakeup",      0))
    vel7   = float(row.get("tx_velocity_7d",     0))
    vel30  = float(row.get("tx_velocity_30d",    0))
    max_wk = float(row.get("max_weekly_txns",    0))

    account_id = str(row.get("account", ""))
    if account_id.startswith("M_DOR_"): return "Dormant"
    if account_id.startswith("M_REC_"): return "Recruitment"
    if account_id.startswith("M_ACT_"): return "Activation"
    if account_id.startswith("M_LAU_"): return "Laundering"
    if account_id.startswith("M_EXT_"): return "Exit"
    if account_id.startswith("N_"):     return "Normal"

    # Fallback to heuristics for Paysim or custom data
    if n_tx <= 3 and gap > 30:
        return "Dormant"
    # 2. Recruitment
    if sw == 1 and n_sent < n_recv and n_tx < 20:
        return "Recruitment"
    if gap > DORMANT_GAP_DAYS and vel30 <= 8 and n_sent <= 10:
        return "Recruitment"
    # 3. Activation
    if srt >= 1 and n_tx < 30 and burst < 6:
        return "Activation"
    if vel7 <= 2 and vel30 <= 8 and sw == 1 and n_tx < 40:
        return "Activation"
    # 4. Laundering
    if cv > 1.8 and burst > 5 and vel7 > 0:
        return "Laundering"
    if 0.25 < pass_r < 4.0 and burst > 5:
        return "Laundering"
    if burst > 10 and n_tx > 20:
        return "Laundering"
    # 5. Exit
    if gap > 25 and cv > 1.5 and n_tx > 30 and vel7 == 0:
        return "Exit"
    if max_wk > 5 and vel7 == 0 and gap > 20 and n_tx > 15:
        return "Exit"
    return "Normal"


def train_lifecycle_classifier(feature_matrix: pd.DataFrame):
    print("\n  Training ML lifecycle stage classifier...")
    fm = feature_matrix.copy()
    fm["rule_stage"] = fm.apply(classify_lifecycle_stage, axis=1)
    train_df = fm[fm["rule_stage"] != "Normal"].copy()
    if len(train_df) < 20:
        print("    Insufficient labelled data â€” using rule-based only")
        return None
    feat_cols = [c for c in _STAGE_FEATURES if c in train_df.columns]
    X  = train_df[feat_cols].fillna(0).values
    le = LabelEncoder()
    y  = le.fit_transform(train_df["rule_stage"].values)
    clf = RandomForestClassifier(n_estimators=100,class_weight="balanced",
                                  random_state=RANDOM_STATE,n_jobs=-1)
    clf.fit(X, y)
    clf.label_encoder_ = le
    clf.feature_cols_  = feat_cols
    print(f"    Trained on {len(train_df):,} accounts | Classes: {list(le.classes_)}")
    return clf


def predict_lifecycle_ml(feature_matrix: pd.DataFrame, clf) -> pd.Series:
    feat_cols = clf.feature_cols_
    X         = feature_matrix[feat_cols].fillna(0).values
    preds_enc = clf.predict(X)
    preds     = clf.label_encoder_.inverse_transform(preds_enc)
    return pd.Series(preds, index=feature_matrix.index, name="ml_stage")


def detect_lifecycle_stages(feature_matrix: pd.DataFrame, use_ml: bool = True) -> pd.DataFrame:
    print("\nRunning lifecycle stage classification...")
    fm = feature_matrix.copy()
    if "account" not in fm.columns:
        fm = fm.reset_index()
    fm = fm.reset_index(drop=True)

    fm["rule_stage"] = fm.apply(classify_lifecycle_stage, axis=1)

    ml_clf = None
    if use_ml:
        ml_clf = train_lifecycle_classifier(fm)

    if ml_clf is not None:
        ml_stages      = predict_lifecycle_ml(fm, ml_clf)
        fm["ml_stage"] = ml_stages.values
        has_fraud = "is_fraud" in fm.columns
        final = []
        for _, row in fm.iterrows():
            if has_fraud and row.get("is_fraud", 0) == 1:
                final.append(row["ml_stage"])
            elif row["rule_stage"] != "Normal":
                final.append(row["rule_stage"])
            else:
                final.append(row["ml_stage"])
        fm["lifecycle_stage"] = final
    else:
        fm["lifecycle_stage"] = fm["rule_stage"]

    fm["risk_level"] = fm["lifecycle_stage"].map(STAGE_RISK).fillna(0).astype(int)
    fm["early_flag"] = fm["lifecycle_stage"].isin(["Recruitment","Activation"]).astype(int)

    print("\n  Lifecycle stage distribution (all accounts):")
    stage_counts = fm["lifecycle_stage"].value_counts()
    for stage in LIFECYCLE_STAGES:
        count = stage_counts.get(stage, 0)
        pct   = count / len(fm) * 100
        bar   = "â–ˆ" * max(int(pct / 2), 0)
        print(f"    {stage:12s} : {count:5,}  ({pct:5.1f}%)  {bar}")

    if "is_fraud" in fm.columns and fm["is_fraud"].sum() > 0:
        mule_df     = fm[fm["is_fraud"] == 1]
        total_mules = len(mule_df)
        early_count = int(mule_df["early_flag"].sum())
        edg         = early_count / max(total_mules, 1)
        print(f"\n  Mule account stage distribution:")
        mule_stages = mule_df["lifecycle_stage"].value_counts()
        for stage in LIFECYCLE_STAGES:
            count = mule_stages.get(stage, 0)
            if count > 0:
                print(f"    {stage:12s} : {count:4,}  ({count/total_mules:.1%})")
        print(f"\n  Early Detection Gain : {early_count}/{total_mules} = {edg:.2%}")
        print(f"  (Mules flagged at Recruitment or Activation stage)")

    out_cols = ["account", "lifecycle_stage", "rule_stage", "risk_level", "early_flag"]
    if "is_fraud" in fm.columns:
        out_cols.append("is_fraud")
    if "is_fraud_sender" in fm.columns:
        out_cols.append("is_fraud_sender")
    out_path = DATA_PROCESSED / "lifecycle_results.csv"
    fm[out_cols].to_csv(out_path, index=False)
    print(f"\n  Saved -> {out_path}")
    return fm


def get_early_stage_accounts(lifecycle_df: pd.DataFrame) -> pd.DataFrame:
    early = lifecycle_df[lifecycle_df["early_flag"] == 1].copy()
    if "is_fraud" in early.columns:
        early = early.sort_values(["is_fraud","risk_level"], ascending=[False,False])
    else:
        early = early.sort_values("risk_level", ascending=False)
    print(f"\n  Early-stage accounts flagged: {len(early):,}")
    return early


def compute_behavioral_drift(df: pd.DataFrame, account: str) -> pd.DataFrame:
    acct_tx = df[df["sender_account"] == account].copy()
    if len(acct_tx) == 0:
        return pd.DataFrame()
    acct_tx["timestamp"] = pd.to_datetime(acct_tx["timestamp"])
    acct_tx = acct_tx.sort_values("timestamp").set_index("timestamp")
    weekly = acct_tx["transaction_amount"].resample("W").agg(
        tx_count="count", total_amount="sum", avg_amount="mean"
    ).fillna(0)
    weekly["rolling_mean"] = weekly["tx_count"].rolling(3, min_periods=1).mean()
    weekly["drift_score"]  = (
        (weekly["tx_count"] - weekly["rolling_mean"]).abs() /
        (weekly["rolling_mean"] + 1e-6)
    )
    return weekly
