"""
src/features/feature_engineering.py
=====================================
Computes a rich feature matrix for every account node.

Feature groups
--------------
A. Node behavioral features   — derived from raw transactions
B. Graph topology features    — derived from NetworkX graph structure
C. Balance/flow features      — derived from amount columns
D. Temporal features          — sliding window behavioral drift

Why each feature indicates mule behaviour
-----------------------------------------
  out_degree            : mules forward money to many accounts (high fan-out)
  in_degree             : mule collectors receive from many sources
  passthrough_ratio     : total_sent / total_recv ≈ 1.0 for pass-through mules
  tx_velocity_24h       : rapid burst sending in short window
  drain_rate            : mules empty accounts after receiving
  pagerank              : mule hubs have high influence in the flow graph
  betweenness           : money passes *through* mule accounts (bridges)
  burst_ratio           : sudden spike in activity vs historical baseline
  small_round_tx_rate   : high rate of test transactions before large transfers
"""

import pandas as pd
import numpy as np
import networkx as nx
from collections import defaultdict
import warnings
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from config.config import *

warnings.filterwarnings("ignore")


# ═══════════════════════════════════════════════════════════════════════════════
# A. NODE BEHAVIORAL FEATURES
# ═══════════════════════════════════════════════════════════════════════════════

def compute_node_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-account transaction statistics from the raw edge list.
    Returns DataFrame indexed by account.
    """
    print("  Computing node behavioral features...")

    # Pre-group for performance
    sent = df.groupby("sender_account")
    recv = df.groupby("receiver_account")

    sent_stats = sent["transaction_amount"].agg(
        total_sent="sum", avg_sent="mean", max_sent="max",
        std_sent="std", n_sent="count"
    ).fillna(0)

    recv_stats = recv["transaction_amount"].agg(
        total_recv="sum", avg_recv="mean", max_recv="max",
        std_recv="std", n_recv="count"
    ).fillna(0)

    # Unique counterparties
    unique_dest   = sent["receiver_account"].nunique().rename("unique_dest")
    unique_source = recv["sender_account"].nunique().rename("unique_source")

    # Balance-drain rate: rows where sender balance goes to ~zero
    # (works if dataset has balance columns; otherwise derived from amounts)
    drain_rate = pd.Series(dtype=float, name="drain_rate")
    if "sender_drained" in df.columns:
        drain_rate = sent["sender_drained"].mean().rename("drain_rate")

    # Pass-through ratio
    all_accounts = pd.Index(pd.unique(
        pd.concat([df["sender_account"], df["receiver_account"]], ignore_index=True)
    ))
    feats = pd.DataFrame(index=all_accounts)
    feats.index.name = "account"

    feats = feats.join(sent_stats,  how="left")
    feats = feats.join(recv_stats,  how="left")
    feats = feats.join(unique_dest, how="left")
    feats = feats.join(unique_source, how="left")
    if len(drain_rate):
        feats = feats.join(drain_rate, how="left")
    else:
        feats["drain_rate"] = 0.0

    feats = feats.fillna(0)

    # Derived
    feats["tx_count"]          = feats["n_sent"] + feats["n_recv"]
    feats["passthrough_ratio"] = feats["total_sent"] / (feats["total_recv"] + 1e-6)
    feats["fanout_ratio"]      = feats["unique_dest"] / (feats["n_sent"] + 1e-6)
    feats["degree_ratio"]      = (feats["n_sent"] + 1) / (feats["n_recv"] + 1)

    # Round-amount rate (test transaction signal)
    round_mask = (
        (df["transaction_amount"] > 0) &
        (np.mod(df["transaction_amount"], TEST_TX_ROUND_MODULO) == 0)
    )
    round_sent = df[round_mask].groupby("sender_account").size().rename("round_tx_count")
    feats = feats.join(round_sent, how="left")
    feats["round_tx_count"] = feats["round_tx_count"].fillna(0)
    feats["small_round_tx_rate"] = feats["round_tx_count"] / (feats["n_sent"] + 1e-6)

    # Cash-out ratio (if type column is available)
    if "type" in df.columns:
        cashout = df[df["type"] == "CASH_OUT"].groupby("sender_account").size()
        feats   = feats.join(cashout.rename("n_cashout"), how="left")
        feats["n_cashout"]    = feats["n_cashout"].fillna(0)
        feats["cashout_rate"] = feats["n_cashout"] / (feats["n_sent"] + 1e-6)
    else:
        feats["cashout_rate"] = 0.0

    if {"origin_balance_error", "dest_balance_error", "balance_error_flag"}.issubset(df.columns):
        sender_balance_error = sent["origin_balance_error"].mean().rename("avg_origin_balance_error")
        receiver_balance_error = recv["dest_balance_error"].mean().rename("avg_dest_balance_error")
        sender_error_flag = sent["balance_error_flag"].mean().rename("sender_balance_error_rate")
        receiver_error_flag = recv["balance_error_flag"].mean().rename("receiver_balance_error_rate")
        feats = feats.join(sender_balance_error, how="left")
        feats = feats.join(receiver_balance_error, how="left")
        feats = feats.join(sender_error_flag, how="left")
        feats = feats.join(receiver_error_flag, how="left")
    else:
        feats["avg_origin_balance_error"] = 0.0
        feats["avg_dest_balance_error"] = 0.0
        feats["sender_balance_error_rate"] = 0.0
        feats["receiver_balance_error_rate"] = 0.0

    if "dest_no_increase" in df.columns:
        feats = feats.join(
            recv["dest_no_increase"].mean().rename("dest_no_increase_rate"),
            how="left"
        )
    else:
        feats["dest_no_increase_rate"] = 0.0

    if "type" in df.columns:
        type_mix = (
            pd.crosstab(df["sender_account"], df["type"], normalize="index")
            .rename(columns=lambda c: f"sender_type_{str(c).lower()}_rate")
        )
        feats = feats.join(type_mix, how="left")

    print(f"    Computed {len(feats.columns)} behavioral features for "
          f"{len(feats):,} accounts")
    return feats.fillna(0)


# ═══════════════════════════════════════════════════════════════════════════════
# B. GRAPH TOPOLOGY FEATURES
# ═══════════════════════════════════════════════════════════════════════════════

def compute_graph_features(G: nx.DiGraph) -> pd.DataFrame:
    """
    Compute graph topology features for every node.
    """
    print("  Computing graph topology features...")

    nodes = list(G.nodes())

    # Degree features
    out_deg = dict(G.out_degree())
    in_deg  = dict(G.in_degree())

    # PageRank — high value = central hub in money flow (mule hubs are high)
    print("    PageRank...", end=" ", flush=True)
    pagerank = nx.pagerank(G, alpha=PAGERANK_ALPHA, max_iter=PAGERANK_MAX_ITER,
                           weight="total_amount")
    print("done")

    # Betweenness centrality (approx) — money passes *through* mule bridges
    betweenness_k = BETWEENNESS_K
    if len(nodes) > LARGE_GRAPH_THRESHOLD:
        betweenness_k = min(LARGE_BETWEENNESS_K, len(nodes))

    print(f"    Betweenness centrality (k={betweenness_k})...", end=" ", flush=True)
    betweenness = nx.betweenness_centrality(
        G, k=min(betweenness_k, len(nodes)), normalized=True, weight="total_amount"
    )
    print("done")

    # Clustering coefficient is one of the slowest metrics on large sparse graphs.
    # For big PaySim runs we skip it so the pipeline can reach model training.
    if len(nodes) > CLUSTERING_SKIP_THRESHOLD:
        print(
            f"    Clustering coefficient skipped (> {CLUSTERING_SKIP_THRESHOLD:,} nodes); "
            "using 0.0 fallback"
        )
        clustering = {n: 0.0 for n in nodes}
    else:
        print("    Clustering coefficient...", end=" ", flush=True)
        G_und = G.to_undirected()
        clustering = nx.clustering(G_und, weight="total_amount")
        print("done")

    # Weakly connected component size — mule clusters form large components
    comp_map = {}
    for comp in nx.weakly_connected_components(G):
        size = len(comp)
        for n in comp:
            comp_map[n] = size

    feats = pd.DataFrame({
        "account":         nodes,
        "out_degree":      [out_deg.get(n, 0) for n in nodes],
        "in_degree":       [in_deg.get(n, 0)  for n in nodes],
        "pagerank":        [pagerank.get(n, 0)     for n in nodes],
        "betweenness":     [betweenness.get(n, 0)  for n in nodes],
        "clustering":      [clustering.get(n, 0)   for n in nodes],
        "component_size":  [comp_map.get(n, 1)     for n in nodes],
    }).set_index("account")

    feats["degree_ratio_graph"] = (feats["out_degree"] + 1) / (feats["in_degree"] + 1)

    print(f"    Computed {len(feats.columns)} topology features for "
          f"{len(feats):,} nodes")
    return feats


# ═══════════════════════════════════════════════════════════════════════════════
# C. TEMPORAL FEATURES (sliding window)
# ═══════════════════════════════════════════════════════════════════════════════

def compute_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute time-based behavioral features:
      - Transaction velocity in 1/7/30-day windows
      - Burst score (peak week / avg week)
      - Max inactivity gap (dormancy signal)
      - Sudden wakeup indicator
      - Days active
    """
    print("  Computing temporal features...")

    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"])

    activity = pd.concat([
        df[["sender_account", "timestamp", "transaction_amount"]].rename(
            columns={"sender_account": "account"}
        ),
        df[["receiver_account", "timestamp", "transaction_amount"]].rename(
            columns={"receiver_account": "account"}
        ),
    ], ignore_index=True)
    activity = activity.sort_values(["account", "timestamp"]).reset_index(drop=True)

    if activity.empty:
        return pd.DataFrame()

    summary = activity.groupby("account").agg(
        first_seen_ts=("timestamp", "min"),
        last_seen_ts=("timestamp", "max"),
    )
    summary["active_days"] = (
        (summary["last_seen_ts"] - summary["first_seen_ts"]).dt.days.clip(lower=1)
    )

    gaps = (
        activity.groupby("account")["timestamp"]
        .diff()
        .dt.total_seconds()
        .div(86400)
    )
    summary["max_gap_days"] = gaps.groupby(activity["account"]).max().fillna(0)

    weekly = (
        activity.assign(week=activity["timestamp"].dt.to_period("W"))
        .groupby(["account", "week"])
        .size()
        .rename("weekly_count")
        .reset_index()
    )
    weekly_stats = weekly.groupby("account")["weekly_count"].agg(
        max_weekly_txns="max",
        mean_weekly_txns="mean",
        weekly_std="std",
    )
    weekly_stats["burst_ratio"] = (
        weekly_stats["max_weekly_txns"] /
        weekly_stats["mean_weekly_txns"].clip(lower=1)
    )
    weekly_stats["weekly_cv"] = (
        weekly_stats["weekly_std"].fillna(0) /
        weekly_stats["mean_weekly_txns"].replace(0, np.nan)
    ).fillna(0)
    weekly_stats = weekly_stats.drop(columns=["weekly_std"])

    sent_activity = df[["sender_account", "timestamp", "transaction_amount"]].rename(
        columns={"sender_account": "account"}
    )
    sent_activity = sent_activity.sort_values(["account", "timestamp"]).reset_index(drop=True)
    sent_activity["ref_ts"] = sent_activity.groupby("account")["timestamp"].transform("max")
    sent_age_days = (
        sent_activity["ref_ts"] - sent_activity["timestamp"]
    ).dt.total_seconds().div(86400)
    sent_activity["vel_1d"] = (sent_age_days <= 1).astype(int)
    sent_activity["vel_7d"] = (sent_age_days <= 7).astype(int)
    sent_activity["vel_30d"] = (sent_age_days <= 30).astype(int)

    velocity = sent_activity.groupby("account").agg(
        tx_velocity_1d=("vel_1d", "sum"),
        tx_velocity_7d=("vel_7d", "sum"),
        tx_velocity_30d=("vel_30d", "sum"),
    )

    first_sent = sent_activity.groupby("account").head(5).copy()
    first_sent["small_round_flag"] = (
        (first_sent["transaction_amount"] < TEST_TX_MAX_AMOUNT) &
        (np.mod(first_sent["transaction_amount"], TEST_TX_ROUND_MODULO) == 0)
    ).astype(int)
    small_round = first_sent.groupby("account")["small_round_flag"].sum().rename("small_round_txns")

    feats = summary.join(weekly_stats, how="left")
    feats = feats.join(velocity, how="left")
    feats = feats.join(small_round, how="left")
    feats["sudden_wakeup"] = (
        (feats["max_gap_days"] > DORMANT_GAP_DAYS) &
        (feats["burst_ratio"] > BURST_RATIO_THRESHOLD)
    ).astype(int)
    feats = feats.fillna(0)
    print(f"    Computed {len(feats.columns)} temporal features for "
          f"{len(feats):,} accounts")
    return feats


# ═══════════════════════════════════════════════════════════════════════════════
# D. ASSEMBLE MASTER FEATURE MATRIX
# ═══════════════════════════════════════════════════════════════════════════════

def build_feature_matrix(
    df: pd.DataFrame,
    G: nx.DiGraph,
    labels: pd.Series = None,
) -> pd.DataFrame:
    """
    Merge all feature groups into a single feature matrix.

    Parameters
    ----------
    df     : canonical transaction DataFrame
    G      : summary weighted DiGraph
    labels : optional Series {account → is_fraud}

    Returns
    -------
    feature_matrix : DataFrame with one row per account, all features + label
    """
    print("\nBuilding master feature matrix...")

    node_feats  = compute_node_features(df)
    graph_feats = compute_graph_features(G)
    temp_feats  = compute_temporal_features(df)

    master = node_feats.join(graph_feats, how="outer", rsuffix="_g")
    master = master.join(temp_feats,       how="outer")
    master = master.fillna(0)

    if labels is not None:
        mapped = labels.reindex(master.index).fillna(0).astype(int)
        master["is_fraud"] = mapped
        # When external labels are passed, we cannot separate sender-only labels.
        master["is_fraud_sender"] = mapped
    elif "is_fraud" in df.columns:
        sender_fraud_accounts = (
            df.loc[df["is_fraud"] == 1, ["sender_account"]]
            .rename(columns={"sender_account": "account"})
            .drop_duplicates()
        )
        sender_fraud_map = (
            sender_fraud_accounts.assign(is_fraud_sender=1)
            .set_index("account")["is_fraud_sender"]
            .reindex(master.index)
            .fillna(0)
            .astype(int)
        )

        fraud_accounts = pd.concat([
            df.loc[df["is_fraud"] == 1, ["sender_account"]].rename(
                columns={"sender_account": "account"}
            ),
            df.loc[df["is_fraud"] == 1, ["receiver_account"]].rename(
                columns={"receiver_account": "account"}
            ),
        ], ignore_index=True)
        fraud_map = (
            fraud_accounts.assign(is_fraud=1)
            .groupby("account")["is_fraud"]
            .max()
            .reindex(master.index)
            .fillna(0)
            .astype(int)
        )
        master["is_fraud_sender"] = sender_fraud_map
        master["is_fraud"] = fraud_map
    else:
        master["is_fraud"] = 0
        master["is_fraud_sender"] = 0

    master.index.name = "account"
    master = master.reset_index()

    print(f"\n  Master feature matrix: {master.shape[0]:,} accounts × "
          f"{master.shape[1]-2} features (+account, +label)")
    print(f"  Fraud accounts in matrix: {master['is_fraud'].sum():,} "
          f"({master['is_fraud'].mean():.2%})")

    # Save
    out_path = DATA_PROCESSED / "feature_matrix.csv"
    master.to_csv(out_path, index=False)
    print(f"  Saved → {out_path}")

    return master


# ── Feature column list (used by models) ──────────────────────────────────────
def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return list of numeric feature columns, excluding id and label."""
    drop = {"account", "is_fraud", "source"}
    return [c for c in df.columns
            if c not in drop and pd.api.types.is_numeric_dtype(df[c])]
