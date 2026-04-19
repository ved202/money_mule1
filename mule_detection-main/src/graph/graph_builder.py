"""
src/graph/graph_builder.py
===========================
Constructs directed transaction graphs from canonical transaction DataFrames.

Graph model:
  Nodes  = bank accounts
  Edges  = directed money flows  (sender → receiver)
  Attrs  = total_amount, tx_count, first_seen, last_seen, is_fraud_edge

Two graph objects are maintained:
  G_full    : multi-edge DiGraph (one edge per transaction — preserves time)
  G_summary : weighted DiGraph  (one edge per sender-receiver pair — for ML)
"""

import networkx as nx
import pandas as pd
import numpy as np
from collections import defaultdict
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from config.config import *


def build_transaction_graph(df: pd.DataFrame) -> tuple[nx.DiGraph, nx.MultiDiGraph]:
    """
    Build both a summary weighted DiGraph and a full MultiDiGraph.

    Parameters
    ----------
    df : canonical transaction DataFrame

    Returns
    -------
    G_summary : nx.DiGraph       — aggregated edge weights
    G_full    : nx.MultiDiGraph  — one edge per individual transaction
    """
    print("Building transaction graphs...")

    # ── Node registry: every unique account ───────────────────────────────────
    senders   = set(df["sender_account"].unique())
    receivers = set(df["receiver_account"].unique())
    all_nodes = senders | receivers

    # Per-node fraud label (account is mule if it ever sent a fraud tx)
    fraud_senders = set(df[df["is_fraud"] == 1]["sender_account"].unique())
    node_labels   = {n: (1 if n in fraud_senders else 0) for n in all_nodes}

    # ── Summary graph (weighted) ───────────────────────────────────────────────
    G_summary = nx.DiGraph()
    G_summary.add_nodes_from([
        (n, {"is_fraud": node_labels[n]}) for n in all_nodes
    ])

    edge_agg = defaultdict(lambda: {
        "total_amount": 0.0, "tx_count": 0,
        "first_seen": None, "last_seen": None,
        "fraud_count": 0,
    })

    for row in df.itertuples(index=False):
        key = (row.sender_account, row.receiver_account)
        e   = edge_agg[key]
        e["total_amount"] += row.transaction_amount
        e["tx_count"]     += 1
        e["fraud_count"]  += int(row.is_fraud)
        ts = row.timestamp
        e["first_seen"] = ts if e["first_seen"] is None else min(e["first_seen"], ts)
        e["last_seen"]  = ts if e["last_seen"]  is None else max(e["last_seen"],  ts)

    for (src, dst), attrs in edge_agg.items():
        if GRAPH_SELF_LOOPS is False and src == dst:
            continue
        attrs["avg_amount"] = attrs["total_amount"] / attrs["tx_count"]
        attrs["is_fraud_edge"] = int(attrs["fraud_count"] > 0)
        G_summary.add_edge(src, dst, **attrs)

    # ── Full multi-edge graph ──────────────────────────────────────────────────
    G_full = nx.MultiDiGraph()
    G_full.add_nodes_from([(n, {"is_fraud": node_labels[n]}) for n in all_nodes])
    for row in df.itertuples(index=False):
        G_full.add_edge(
            row.sender_account,
            row.receiver_account,
            amount=row.transaction_amount,
            timestamp=row.timestamp,
            is_fraud=int(row.is_fraud),
        )

    print(f"  G_summary : {G_summary.number_of_nodes():,} nodes | "
          f"{G_summary.number_of_edges():,} edges")
    print(f"  G_full    : {G_full.number_of_nodes():,} nodes | "
          f"{G_full.number_of_edges():,} edges")
    fraud_edge_pct = sum(
        1 for _, _, d in G_summary.edges(data=True) if d.get("is_fraud_edge")
    ) / max(G_summary.number_of_edges(), 1)
    print(f"  Fraud edge rate: {fraud_edge_pct:.2%}")

    return G_summary, G_full


def get_subgraph(G: nx.DiGraph, nodes: list) -> nx.DiGraph:
    """Extract a subgraph containing given nodes plus their immediate neighbours."""
    neighbourhood = set(nodes)
    for n in nodes:
        neighbourhood.update(G.predecessors(n))
        neighbourhood.update(G.successors(n))
    return G.subgraph(neighbourhood).copy()


def graph_summary(G: nx.DiGraph) -> dict:
    """Return a dict of basic graph statistics."""
    is_connected = nx.is_weakly_connected(G)
    components   = list(nx.weakly_connected_components(G))
    return {
        "n_nodes":        G.number_of_nodes(),
        "n_edges":        G.number_of_edges(),
        "density":        nx.density(G),
        "n_components":   len(components),
        "largest_comp":   max(len(c) for c in components),
        "is_connected":   is_connected,
        "avg_out_degree": np.mean([d for _, d in G.out_degree()]),
        "avg_in_degree":  np.mean([d for _, d in G.in_degree()]),
    }
