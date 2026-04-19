"""
src/community/community_detector.py
=====================================
Detects coordinated mule networks as graph communities.

Methods
-------
1. Louvain community detection  — finds densely connected clusters
2. Weakly connected components  — finds all reachable account groups
3. Suspicious cluster scoring   — ranks communities by mule density

Why mule networks form clusters
--------------------------------
Money mules operate in coordinated networks. Within a network:
  - Collectors receive from the same fraud sources
  - Distributors are all connected to the same collectors
  - The network has higher internal transaction density than random accounts

This creates a detectable "community structure" in the graph where
mule accounts appear in the same dense cluster.
"""

import networkx as nx
import pandas as pd
import numpy as np
from collections import defaultdict
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from config.config import *

# Optional: python-louvain package
try:
    import community as community_louvain
    LOUVAIN_AVAILABLE = True
except ImportError:
    LOUVAIN_AVAILABLE = False
    print("  [Warning] python-louvain not installed. "
          "Install with: pip install python-louvain")
    print("  Falling back to greedy modularity communities.")


def detect_communities(
    G: nx.DiGraph,
    method: str = "louvain",
) -> dict[str, int]:
    """
    Assign each node to a community ID.

    Parameters
    ----------
    G      : directed transaction graph
    method : 'louvain' | 'components' | 'greedy'

    Returns
    -------
    partition : {account → community_id}
    """
    print(f"\nDetecting communities (method={method})...")
    G_und = G.to_undirected()

    if method == "louvain":
        if LOUVAIN_AVAILABLE:
            partition = community_louvain.best_partition(
                G_und, resolution=LOUVAIN_RESOLUTION, random_state=RANDOM_STATE
            )
        else:
            method = "greedy"

    if method == "greedy":
        communities = nx.community.greedy_modularity_communities(G_und)
        partition   = {}
        for cid, comm in enumerate(communities):
            for node in comm:
                partition[node] = cid

    if method == "components":
        partition = {}
        for cid, comp in enumerate(nx.weakly_connected_components(G)):
            for node in comp:
                partition[node] = cid

    n_communities = len(set(partition.values()))
    print(f"  Found {n_communities:,} communities")
    return partition


def score_communities(
    partition: dict[str, int],
    feature_matrix: pd.DataFrame,
    G: nx.DiGraph,
) -> pd.DataFrame:
    """
    Score each community on multiple suspicion indicators.

    Suspicion indicators:
      mule_rate         — fraction of labelled mules in community
      avg_pagerank      — mean PageRank (hub centrality)
      avg_betweenness   — mean betweenness centrality
      avg_passthrough   — mean pass-through ratio
      avg_burst_ratio   — mean burst score
      internal_density  — edges within community / possible edges
      size              — number of accounts

    Returns DataFrame of communities sorted by suspicion_score.
    """
    print("  Scoring community suspicion levels...")

    # Map accounts to features
    fm = feature_matrix.copy()
    if "account" not in fm.columns:
        fm = fm.reset_index()

    fm["community_id"] = fm["account"].map(partition)
    fm = fm.dropna(subset=["community_id"]).copy()
    fm["community_id"] = fm["community_id"].astype(int)

    community_records = []

    # Group accounts by community
    comm_to_nodes: dict[int, list] = defaultdict(list)
    for node, cid in partition.items():
        comm_to_nodes[cid].append(node)

    agg_spec = {"account": "size"}
    for col in [
        "is_fraud", "pagerank", "betweenness",
        "passthrough_ratio", "burst_ratio", "tx_velocity_7d",
    ]:
        if col in fm.columns:
            agg_spec[col] = "mean"

    grouped = fm.groupby("community_id").agg(agg_spec).rename(columns={
        "account": "size",
        "is_fraud": "mule_rate",
        "pagerank": "avg_pagerank",
        "betweenness": "avg_betweenness",
        "passthrough_ratio": "avg_passthrough",
        "burst_ratio": "avg_burst_ratio",
        "tx_velocity_7d": "avg_velocity_7d",
    })

    grouped["accounts"] = grouped.index.map(comm_to_nodes.get)
    grouped = grouped[grouped["size"] >= MIN_COMMUNITY_SIZE].copy()

    if grouped.empty:
        print("  Communities analysed  : 0")
        print("  Flagged as suspicious : 0")
        return pd.DataFrame()

    for cid, row in grouped.iterrows():
        nodes = row["accounts"]
        n = int(row["size"])
        mule_rate = float(row.get("mule_rate", 0.0))
        avg_pr = float(row.get("avg_pagerank", 0.0))
        avg_btw = float(row.get("avg_betweenness", 0.0))
        avg_pass = float(row.get("avg_passthrough", 0.0))
        avg_burst = float(row.get("avg_burst_ratio", 0.0))
        avg_velocity = float(row.get("avg_velocity_7d", 0.0))

        # Internal edge density
        subgraph = G.subgraph(nodes)
        max_edges = n * (n - 1)
        density = subgraph.number_of_edges() / max_edges if max_edges > 0 else 0

        # Total flow through community
        total_flow = sum(
            d.get("total_amount", 0)
            for _, _, d in subgraph.edges(data=True)
        )

        # Composite suspicion score (weighted combination)
        # For mixed communities, balance fraud presence with network behavior
        if mule_rate == 1.0:
            # Pure fraud communities: high but capped suspicion
            suspicion = min(0.8, 0.4 * mule_rate + 0.3 * min(avg_pass, 1.0) + 0.3 * min(density * 5, 1.0))
        elif mule_rate == 0.0:
            # Pure normal communities: low suspicion
            suspicion = 0.1 * min(avg_pr * 1000, 1.0) + 0.1 * min(avg_btw * 50, 1.0)
        else:
            # Mixed communities: balance fraud with coordination signals
            suspicion = (
                0.25 * mule_rate +                    # Fraud presence
                0.20 * min(avg_pr * 100, 1.0) +      # Centrality
                0.15 * min(avg_btw * 10, 1.0) +      # Bridging
                0.15 * min(avg_pass, 1.0) +          # Pass-through behavior
                0.15 * min(avg_burst / 10.0, 1.0) +  # Burst activity
                0.10 * min(density * 10, 1.0)        # Internal connectivity
            )

        community_records.append({
            "community_id":    cid,
            "size":            n,
            "mule_rate":       round(mule_rate, 4),
            "avg_pagerank":    round(avg_pr, 6),
            "avg_betweenness": round(avg_btw, 6),
            "avg_passthrough": round(avg_pass, 4),
            "avg_burst_ratio": round(avg_burst, 3),
            "avg_velocity_7d": round(avg_velocity, 2),
            "internal_density":round(density, 4),
            "total_flow":      round(total_flow, 2),
            "suspicion_score": round(suspicion, 4),
            "is_suspicious":   int(
                (suspicion > 0.35) or  # Higher threshold for mixed communities
                (mule_rate > SUSPICIOUS_COMMUNITY_MULE_RATE and suspicion > 0.25)
            ),
            "accounts":        nodes,
        })

    comm_df = pd.DataFrame(community_records)
    comm_df = comm_df.sort_values("suspicion_score", ascending=False).reset_index(drop=True)

    n_suspicious = comm_df["is_suspicious"].sum()
    print(f"  Communities analysed  : {len(comm_df):,}")
    print(f"  Flagged as suspicious : {n_suspicious:,}")

    return comm_df


def get_suspicious_accounts(
    partition: dict[str, int],
    comm_df: pd.DataFrame,
) -> list[str]:
    """Return list of accounts belonging to suspicious communities."""
    suspicious_cids = set(
        comm_df[comm_df["is_suspicious"] == 1]["community_id"].tolist()
    )
    return [acc for acc, cid in partition.items() if cid in suspicious_cids]
