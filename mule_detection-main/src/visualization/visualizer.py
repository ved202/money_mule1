"""
src/visualization/visualizer.py
================================
All visualisation functions for the mule detection system.

Plots produced
--------------
1.  Full transaction network (coloured by fraud label)
2.  Suspicious mule cluster subgraph
3.  Feature importance bar chart
4.  Precision-Recall curves (all models)
5.  Model comparison bar chart
6.  Lifecycle stage distribution
7.  Temporal drift for a sample mule account
8.  Community suspicion scores
9.  Fund flow heatmap (sender â†’ receiver amount matrix)
10. Anomaly score distribution
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import networkx as nx
import pandas as pd
import numpy as np
from sklearn.metrics import precision_recall_curve, average_precision_score
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from config.config import *


# â”€â”€ Helpers â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def _save(fig, name: str):
    path = OUTPUTS_PLOTS / f"{name}.png"
    fig.savefig(path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved â†’ {path}")


def _node_colors(G: nx.DiGraph, community_map: dict = None) -> list:
    colors = []
    for n in G.nodes():
        label = G.nodes[n].get("is_fraud", 0)
        if label == 1:
            colors.append(NODE_COLOR_MULE)
        elif community_map and community_map.get(n, -1) >= 0:
            colors.append(NODE_COLOR_SUSPECTED)
        else:
            colors.append(NODE_COLOR_NORMAL)
    return colors


# â”€â”€ 1. Full network overview â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def plot_transaction_network(
    G: nx.DiGraph,
    max_nodes: int = 300,
    title: str = "Transaction Network",
    save: bool = True,
):
    """Plot a sample of the transaction graph."""
    print(f"\nPlotting transaction network (max {max_nodes} nodes)...")

    if G.number_of_nodes() > max_nodes:
        # Sample: enforce 60% fraud and 40% normal where possible.
        fraud_nodes  = [n for n, d in G.nodes(data=True) if d.get("is_fraud")]
        normal_nodes = [n for n, d in G.nodes(data=True) if not d.get("is_fraud")]
        rng          = np.random.default_rng(RANDOM_STATE)

        target_fraud  = int(round(max_nodes * 0.60))
        target_normal = max_nodes - target_fraud

        n_fraud       = min(len(fraud_nodes), target_fraud)
        n_normal      = min(len(normal_nodes), target_normal)

        # Fill any remaining slots from whichever class still has capacity.
        remaining     = max_nodes - (n_fraud + n_normal)
        if remaining > 0:
            extra_fraud_cap  = max(len(fraud_nodes) - n_fraud, 0)
            extra_normal_cap = max(len(normal_nodes) - n_normal, 0)

            add_fraud  = min(remaining, extra_fraud_cap)
            n_fraud   += add_fraud
            remaining -= add_fraud

            if remaining > 0:
                n_normal += min(remaining, extra_normal_cap)

        sample_fraud = (
            list(rng.choice(fraud_nodes, n_fraud, replace=False))
            if n_fraud > 0 else []
        )
        sample_norm = (
            list(rng.choice(normal_nodes, n_normal, replace=False))
            if n_normal > 0 else []
        )
        keep         = set(sample_fraud + sample_norm)
        G            = G.subgraph(keep).copy()

    fig, ax = plt.subplots(figsize=PLOT_FIGSIZE_LARGE)
    pos = nx.spring_layout(G, seed=RANDOM_STATE, k=0.8)

    colors   = _node_colors(G)
    sizes    = [120 if G.nodes[n].get("is_fraud") else 120 for n in G.nodes()]
    weights  = [d.get("total_amount", 1) for _, _, d in G.edges(data=True)]
    max_w    = max(weights) if weights else 1
    widths   = [0.3 + 2.0 * (w / max_w) for w in weights]

    nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.25,
                           width=widths, edge_color=EDGE_COLOR_NORMAL,
                           arrows=True, arrowsize=8)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=colors,
                           node_size=sizes, alpha=0.85)

    # Legend
    legend_items = [
        mpatches.Patch(color=NODE_COLOR_MULE,   label="Mule account"),
        mpatches.Patch(color=NODE_COLOR_NORMAL, label="Normal account"),
    ]
    ax.legend(handles=legend_items, loc="upper left", fontsize=10, frameon=False)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.axis("off")
    plt.tight_layout()

    if save:
        _save(fig, "01_transaction_network")
    return fig


# â”€â”€ 2. Suspicious cluster subgraph â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def plot_mule_cluster(
    G: nx.DiGraph,
    cluster_nodes: list,
    title: str = "Detected Mule Cluster",
    save: bool = True,
):
    """Visualise a single detected mule network community."""
    print(f"\nPlotting mule cluster ({len(cluster_nodes)} accounts)...")

    # Include 1-hop neighbours for context
    neighbourhood = set(cluster_nodes)
    for n in cluster_nodes:
        neighbourhood.update(list(G.predecessors(n))[:3])
        neighbourhood.update(list(G.successors(n))[:3])

    sub = G.subgraph(neighbourhood).copy()

    fig, ax = plt.subplots(figsize=PLOT_FIGSIZE_MEDIUM)
    pos = nx.spring_layout(sub, seed=RANDOM_STATE, k=1.2)

    # Colour: red = confirmed mule, orange = in cluster but unknown, blue = neighbour
    cluster_set = set(cluster_nodes)
    node_colors = []
    node_sizes  = []
    for n in sub.nodes():
        is_m = sub.nodes[n].get("is_fraud", 0)
        if is_m:
            node_colors.append(NODE_COLOR_MULE);      node_sizes.append(200)
        elif n in cluster_set:
            node_colors.append(NODE_COLOR_SUSPECTED); node_sizes.append(150)
        else:
            node_colors.append(NODE_COLOR_NORMAL);    node_sizes.append(60)

    edge_colors = [
        EDGE_COLOR_SUSPICIOUS if (
            sub.nodes[u].get("is_fraud") or sub.nodes[v].get("is_fraud")
        ) else EDGE_COLOR_NORMAL
        for u, v in sub.edges()
    ]

    nx.draw_networkx_edges(sub, pos, ax=ax, edge_color=edge_colors,
                           alpha=0.5, arrows=True, arrowsize=10)
    nx.draw_networkx_nodes(sub, pos, ax=ax, node_color=node_colors,
                           node_size=node_sizes, alpha=0.9)
    nx.draw_networkx_labels(sub, pos, ax=ax,
                            labels={n: n[-6:] for n in sub.nodes()},
                            font_size=6, font_color="#2C2C2A")

    legend_items = [
        mpatches.Patch(color=NODE_COLOR_MULE,      label="Confirmed mule"),
        mpatches.Patch(color=NODE_COLOR_SUSPECTED, label="Suspected (in cluster)"),
        mpatches.Patch(color=NODE_COLOR_NORMAL,    label="Neighbouring account"),
    ]
    ax.legend(handles=legend_items, loc="upper left", fontsize=9, frameon=False)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.axis("off")
    plt.tight_layout()

    if save:
        _save(fig, "02_mule_cluster")
    return fig


# â”€â”€ 3. Feature importance â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def plot_feature_importance(
    feat_importance: pd.Series,
    top_n: int = 20,
    save: bool = True,
):
    print("\nPlotting feature importances...")
    fi = feat_importance.sort_values(ascending=True).tail(top_n)

    fig, ax = plt.subplots(figsize=(9, 7))
    colors  = ["#185FA5" if v > fi.median() else "#85B7EB" for v in fi.values]
    fi.plot(kind="barh", ax=ax, color=colors, edgecolor="none", alpha=0.9)

    ax.set_title(f"Top {top_n} Feature Importances â€” Random Forest",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Importance score")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()

    if save:
        _save(fig, "03_feature_importance")
    return fig


# â”€â”€ 4. Precision-Recall curves â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def plot_pr_curves(
    model_results: dict,   # {model_name: {"y_test": ..., "y_prob_test": ...}}
    save: bool = True,
):
    print("\nPlotting Precision-Recall curves...")
    colors = {"IsolationForest": "#888780", "RandomForest": "#185FA5",
              "GradientBoosting": "#1D9E75", "Ensemble": "#E24B4A"}

    fig, ax = plt.subplots(figsize=PLOT_FIGSIZE_SMALL)
    for name, res in model_results.items():
        y_true = res.get("y_test")
        y_prob = res.get("y_prob_test")
        if y_true is None or y_prob is None:
            continue
        p, r, _ = precision_recall_curve(y_true, y_prob)
        ap      = average_precision_score(y_true, y_prob)
        ax.plot(r, p, label=f"{name} (AP={ap:.3f})",
                color=colors.get(name, "#888780"), lw=2)

    ax.set_xlabel("Recall", fontsize=11)
    ax.set_ylabel("Precision", fontsize=11)
    ax.set_title("Precision-Recall Curves â€” All Models", fontsize=13, fontweight="bold")
    ax.legend(frameon=False, fontsize=10)
    ax.grid(True, alpha=0.2)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()

    if save:
        _save(fig, "04_pr_curves")
    return fig


# â”€â”€ 5. Model comparison bar chart â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def plot_model_comparison(
    comparison_df: pd.DataFrame,
    save: bool = True,
):
    print("\nPlotting model comparison...")
    metrics = ["Precision", "Recall", "F1", "PR-AUC"]
    models  = comparison_df["Model"].tolist()
    colors  = ["#B4B2A9", "#185FA5", "#1D9E75", "#E24B4A"][:len(models)]

    x     = np.arange(len(metrics))
    width = 0.8 / len(models)

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, (model, color) in enumerate(zip(models, colors)):
        row    = comparison_df[comparison_df["Model"] == model].iloc[0]
        vals   = [row.get(m, 0) for m in metrics]
        offset = (i - len(models) / 2 + 0.5) * width
        bars   = ax.bar(x + offset, vals, width, label=model,
                        color=color, alpha=0.88, edgecolor="none")
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.005,
                    f"{v:.2f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=11)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison â€” Mule Detection Performance",
                 fontsize=13, fontweight="bold")
    ax.legend(frameon=False, fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()

    if save:
        _save(fig, "05_model_comparison")
    return fig


# â”€â”€ 6. Lifecycle distribution â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def plot_lifecycle_distribution(lifecycle_df: pd.DataFrame, save: bool = True):
    print("\nPlotting lifecycle stage distribution...")
    stage_order  = ["Dormant", "Recruitment", "Activation", "Laundering", "Exit", "Normal"]
    stage_colors = ["#B4B2A9", "#85B7EB", "#EF9F27", "#E24B4A", "#888780", "#9FE1CB"]

    if "is_fraud" in lifecycle_df.columns:
        mule_counts = lifecycle_df[lifecycle_df["is_fraud"] == 1][
            "lifecycle_stage"
        ].value_counts().reindex(stage_order, fill_value=0)
    else:
        mule_counts = pd.Series(0, index=stage_order)

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    bars = ax.bar(stage_order, mule_counts.values, color=stage_colors, edgecolor="none", alpha=0.9)
    ax.set_title("Confirmed mule accounts", fontsize=12, fontweight="bold")
    ax.set_ylabel("Number of accounts")
    ax.spines[["top", "right"]].set_visible(False)
    for bar, v in zip(bars, mule_counts.values):
        if v > 0:
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.3, str(v),
                    ha="center", va="bottom", fontsize=10)
    ax.tick_params(axis="x", rotation=20)
    plt.suptitle("Mule Lifecycle Stage Distribution", fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save:
        _save(fig, "06_lifecycle_distribution")
    return fig


# â”€â”€ 7. Temporal drift for a sample account â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def plot_temporal_drift(
    df: pd.DataFrame,
    mule_account: str,
    normal_account: str,
    save: bool = True,
):
    print("\nPlotting temporal drift example...")

    def daily_series(acc):
        tx = df[df["sender_account"] == acc].copy()
        if tx.empty:
            return pd.Series(dtype=float)
        tx["timestamp"] = pd.to_datetime(tx["timestamp"], errors="coerce")
        tx = tx.dropna(subset=["timestamp"]).set_index("timestamp").sort_index()
        daily = tx["transaction_amount"].resample("D").size()
        daily = daily[daily > 0]
        return daily

    mule_wk   = daily_series(mule_account)
    normal_wk = daily_series(normal_account)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, series, title, color in [
        (axes[0], mule_wk,   f"Mule: {mule_account[:10]}",   "#E24B4A"),
        (axes[1], normal_wk, f"Normal: {normal_account[:10]}", "#185FA5"),
    ]:
        if len(series) == 0:
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center")
        else:
            x = series.index.to_pydatetime()
            ax.bar(x, series.values, color=color, alpha=0.82,
                   edgecolor="none", width=0.8)
            ax.axhline(series.mean(), color="#444441", linestyle="--",
                       linewidth=1, label=f"Mean={series.mean():.1f}")
            ax.xaxis.set_major_locator(mdates.AutoDateLocator())
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
            ax.tick_params(axis="x", rotation=35)
            ax.legend(frameon=False, fontsize=9)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel("Date")
        ax.set_ylabel("Transactions sent")
        ax.spines[["top", "right"]].set_visible(False)

    plt.suptitle("Behavioral Drift: Mule vs Normal Account", fontsize=13, fontweight="bold")
    plt.tight_layout()

    if save:
        _save(fig, "07_temporal_drift")
    return fig


# â”€â”€ 8. Community suspicion scores â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def plot_community_suspicion(comm_df: pd.DataFrame, save: bool = True):
    print("\nPlotting community suspicion scores...")
    
    # Filter for more meaningful communities: size >= 5 and not 100% fraud
    meaningful = comm_df[
        (comm_df["size"] >= 5) & 
        (comm_df["mule_rate"] < 1.0) &  # Include some normal accounts
        (comm_df["mule_rate"] > 0.0)   # Include some fraud accounts
    ].head(20)
    
    if len(meaningful) == 0:
        # Fallback: just show largest communities with mixed account types
        meaningful = comm_df[comm_df["size"] >= 5].head(20)
    
    if len(meaningful) == 0:
        # Final fallback: original behavior
        meaningful = comm_df.head(20)
    
    top = meaningful.copy()
    top["label"] = [f"C{int(r.community_id)} (n={int(r['size'])})"
                    for _, r in top.iterrows()]

    fig, ax = plt.subplots(figsize=(9, 6))
    colors  = [NODE_COLOR_MULE if row["is_suspicious"] else NODE_COLOR_NORMAL
               for _, row in top.iterrows()]
    ax.barh(top["label"], top["suspicion_score"], color=colors, alpha=0.88, edgecolor="none")
    ax.axvline(0.25, color="#444441", linestyle="--", linewidth=1, label="Alert threshold")
    ax.set_xlabel("Suspicion score")
    ax.set_title(f"Top {len(top)} Mixed Communities by Suspicion Score", fontsize=13, fontweight="bold")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()

    if save:
        _save(fig, "08_community_suspicion")
    return fig


# â”€â”€ 9. Anomaly score distribution â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def plot_anomaly_scores(
    scores: pd.Series,
    labels: pd.Series = None,
    threshold: float = None,
    save: bool = True,
):
    print("\nPlotting anomaly score distribution...")
    fig, ax = plt.subplots(figsize=PLOT_FIGSIZE_SMALL)

    # Filter out NaN values
    scores = scores.dropna()
    if len(scores) == 0:
        ax.text(0.5, 0.5, "No valid scores to plot", 
                transform=ax.transAxes, ha="center", va="center")
        ax.set_title("Anomaly Score Distribution (No Data)", fontsize=13, fontweight="bold")
        if save:
            _save(fig, "09_anomaly_distribution")
        return fig

    if labels is not None and labels.sum() > 0:
        common = scores.index.intersection(labels.index)
        if len(common) == 0:
            # No matching labels, fall back to single histogram
            ax.hist(scores.values, bins=min(60, len(scores)//10 + 1), alpha=0.8,
                    color=NODE_COLOR_NORMAL, density=False)
            ax.set_ylabel("Count")
        else:
            s = scores.loc[common]; l = labels.loc[common]
            ax.hist(s[l == 0].values, bins=min(60, len(s)//20 + 1), alpha=0.6,
                    color=NODE_COLOR_NORMAL, label="Normal", density=False)
            ax.hist(s[l == 1].values, bins=min(60, len(s)//20 + 1), alpha=0.7,
                    color=NODE_COLOR_MULE, label="Mule", density=False)
            ax.legend(frameon=False, fontsize=10)
            ax.set_ylabel("Count")
    else:
        ax.hist(scores.values, bins=min(60, len(scores)//10 + 1), 
                color=NODE_COLOR_NORMAL, alpha=0.8, density=False)
        ax.set_ylabel("Count")

    if threshold is not None:
        ax.axvline(threshold, color="#E24B4A", linestyle="--",
                   linewidth=1.5, label=f"Threshold={threshold:.3f}")
        ax.legend(frameon=False, fontsize=10)

    ax.set_xlabel("Anomaly / suspicion score")
    ax.set_title("Anomaly Score Distribution", fontsize=13, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()

    if save:
        _save(fig, "09_anomaly_distribution")
    return fig
