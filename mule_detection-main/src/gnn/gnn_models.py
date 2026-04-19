"""
src/gnn/gnn_models.py
======================
Graph Neural Network models for mule account detection.

Why GNNs outperform classical ML on this problem
-------------------------------------------------
Classical ML (Random Forest, Gradient Boosting) uses only the features of
each account in isolation. GNNs additionally consider the NEIGHBOURHOOD:
a mule account surrounded by other mules looks different from an identical
account surrounded by normal accounts.

This is called "guilt by association" — the graph structure itself is a signal.

Two architectures implemented
------------------------------
1. GraphSAGE (Hamilton et al. 2017)
   - Aggregates fixed-size neighbourhoods via sampling
   - Fast, scalable to millions of nodes
   - Uses mean/max/LSTM aggregators
   - Best for: large graphs where full neighbourhood is expensive

2. GAT — Graph Attention Network (Veličković et al. 2018)
   - Learns attention weights: WHICH neighbours matter most
   - A mule's suspicious neighbours get higher attention weight
   - More expressive but slower than SAGE
   - Best for: medium graphs where neighbour quality varies

Training strategy
-----------------
- Semi-supervised: use labelled fraud accounts + unlabelled accounts
- Class-weighted cross-entropy for extreme imbalance (mules are rare)
- Early stopping on validation PR-AUC (more meaningful than accuracy)
- Final predictions include embedding vectors for visualisation

Requirements
------------
  pip install torch torch_geometric

If torch is unavailable, this module degrades gracefully and the pipeline
continues using classical ML from ml_models.py.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from config.config import *

# ── Torch availability guard ──────────────────────────────────────────────────
try:
    import torch
    import torch.nn.functional as F
    from torch_geometric.data import Data
    from torch_geometric.nn import SAGEConv, GATConv, BatchNorm
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score, roc_auc_score, classification_report
import warnings
warnings.filterwarnings("ignore")


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL ARCHITECTURES
# ═══════════════════════════════════════════════════════════════════════════════

if TORCH_AVAILABLE:

    class GraphSAGEDetector(torch.nn.Module):
        """
        3-layer GraphSAGE for mule detection.

        Architecture:
            Input (F features)
            → SAGEConv(F, 128) + BatchNorm + ReLU + Dropout
            → SAGEConv(128, 64) + BatchNorm + ReLU + Dropout
            → SAGEConv(64, 32)  + ReLU
            → Linear(32, 2)     → class logits

        Each SAGEConv layer:
            h_v = W · CONCAT(h_v, MEAN({h_u : u ∈ N(v)}))
        """

        def __init__(self, in_channels: int, hidden1: int = 128,
                     hidden2: int = 64, hidden3: int = 32,
                     dropout: float = 0.4):
            super().__init__()
            self.conv1 = SAGEConv(in_channels, hidden1)
            self.bn1   = BatchNorm(hidden1)
            self.conv2 = SAGEConv(hidden1, hidden2)
            self.bn2   = BatchNorm(hidden2)
            self.conv3 = SAGEConv(hidden2, hidden3)
            self.head  = torch.nn.Linear(hidden3, 2)
            self.drop  = dropout

        def forward(self, x, edge_index):
            # Layer 1
            x = self.conv1(x, edge_index)
            x = self.bn1(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.drop, training=self.training)
            # Layer 2
            x = self.conv2(x, edge_index)
            x = self.bn2(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.drop, training=self.training)
            # Layer 3
            x = self.conv3(x, edge_index)
            x = F.relu(x)
            embedding = x                        # save for visualisation
            logits    = self.head(x)
            return logits, embedding

        def predict_proba(self, x, edge_index):
            self.eval()
            with torch.no_grad():
                logits, emb = self.forward(x, edge_index)
                return F.softmax(logits, dim=1)[:, 1], emb


    class GATDetector(torch.nn.Module):
        """
        2-layer Graph Attention Network for mule detection.

        Architecture:
            Input (F features)
            → GATConv(F, 64, heads=8, concat=True)  → 512 dim
            → ELU + Dropout
            → GATConv(512, 32, heads=4, concat=True) → 128 dim
            → ELU + Dropout
            → Linear(128, 2) → class logits

        Each GATConv layer learns:
            α_ij = softmax( LeakyReLU( a^T [W h_i || W h_j] ) )
            h'_i  = σ( Σ_j α_ij W h_j )

        The attention weights α_ij tell us which neighbours
        most influence each account's classification.
        """

        def __init__(self, in_channels: int, hidden: int = 64,
                     heads1: int = 8, heads2: int = 4, dropout: float = 0.4):
            super().__init__()
            self.conv1 = GATConv(in_channels, hidden, heads=heads1,
                                  dropout=dropout, concat=True)
            self.conv2 = GATConv(hidden * heads1, hidden // 2, heads=heads2,
                                  dropout=dropout, concat=True)
            self.head  = torch.nn.Linear(hidden // 2 * heads2, 2)
            self.drop  = dropout

        def forward(self, x, edge_index, return_attention: bool = False):
            x = F.dropout(x, p=self.drop, training=self.training)
            if return_attention:
                x, (edge_idx, attn_w) = self.conv1(x, edge_index,
                                                     return_attention_weights=True)
            else:
                x = self.conv1(x, edge_index)
                attn_w = None
            x = F.elu(x)
            x = F.dropout(x, p=self.drop, training=self.training)
            x = self.conv2(x, edge_index)
            x = F.elu(x)
            embedding = x
            logits    = self.head(x)
            return logits, embedding

        def predict_proba(self, x, edge_index):
            self.eval()
            with torch.no_grad():
                logits, emb = self.forward(x, edge_index)
                return F.softmax(logits, dim=1)[:, 1], emb


# ═══════════════════════════════════════════════════════════════════════════════
# DATA PREPARATION
# ═══════════════════════════════════════════════════════════════════════════════

def prepare_pyg_data(
    feature_matrix: pd.DataFrame,
    df_transactions: pd.DataFrame,
    feat_cols: list[str],
) -> "Data":
    """
    Convert feature matrix + transaction DataFrame into a
    PyTorch Geometric Data object.

    Returns
    -------
    data        : PyG Data(x, edge_index, y, train_mask, val_mask, test_mask)
    scaler      : fitted StandardScaler (for inference)
    account_idx : {account_name → integer_node_id}
    """
    if not TORCH_AVAILABLE:
        raise ImportError("torch and torch_geometric are required for GNN training.")

    # ── Node ordering: sort feature matrix by account name ────────────────────
    fm = feature_matrix.copy()
    if "account" not in fm.columns:
        fm = fm.reset_index()

    fm          = fm.sort_values("account").reset_index(drop=True)
    account_idx = {acc: i for i, acc in enumerate(fm["account"].tolist())}

    # ── Node feature matrix ───────────────────────────────────────────────────
    available   = [c for c in feat_cols if c in fm.columns]
    scaler      = StandardScaler()
    X_np        = scaler.fit_transform(fm[available].fillna(0).values.astype(np.float32))
    x           = torch.tensor(X_np, dtype=torch.float)

    # ── Labels ────────────────────────────────────────────────────────────────
    y_np = fm["is_fraud"].fillna(0).astype(int).values
    y    = torch.tensor(y_np, dtype=torch.long)

    # ── Edge index (sender → receiver) ───────────────────────────────────────
    src_ids, dst_ids = [], []
    for row in df_transactions.itertuples(index=False):
        s = account_idx.get(row.sender_account)
        d = account_idx.get(row.receiver_account)
        if s is not None and d is not None:
            src_ids.append(s)
            dst_ids.append(d)

    edge_index = torch.tensor([src_ids, dst_ids], dtype=torch.long)

    # ── Train / val / test masks (prefer stratified when labels permit) ──────
    n = len(fm)
    all_idx = np.arange(n)
    y_np = y.numpy()

    if len(np.unique(y_np)) > 1 and y_np.sum() >= 3 and (y_np == 0).sum() >= 3:
        tr_idx, holdout_idx = train_test_split(
            all_idx,
            test_size=0.30,
            random_state=RANDOM_STATE,
            stratify=y_np,
        )
        holdout_y = y_np[holdout_idx]
        stratify_holdout = holdout_y if len(np.unique(holdout_y)) > 1 else None
        val_idx, test_idx = train_test_split(
            holdout_idx,
            test_size=0.50,
            random_state=RANDOM_STATE,
            stratify=stratify_holdout,
        )
    else:
        rng = np.random.default_rng(RANDOM_STATE)
        perm = rng.permutation(n)
        tr_end = int(0.70 * n)
        val_end = int(0.85 * n)
        tr_idx = perm[:tr_end]
        val_idx = perm[tr_end:val_end]
        test_idx = perm[val_end:]

    train_mask = torch.zeros(n, dtype=torch.bool)
    val_mask = torch.zeros(n, dtype=torch.bool)
    test_mask = torch.zeros(n, dtype=torch.bool)
    train_mask[tr_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True

    data = Data(
        x=x, edge_index=edge_index, y=y,
        train_mask=train_mask, val_mask=val_mask, test_mask=test_mask,
    )

    print(f"\n  PyG graph built:")
    print(f"    Nodes      : {data.num_nodes:,}")
    print(f"    Edges      : {data.num_edges:,}")
    print(f"    Features   : {data.num_node_features}")
    print(f"    Train/Val/Test: {train_mask.sum()}/{val_mask.sum()}/{test_mask.sum()}")

    return data, scaler, account_idx


# ═══════════════════════════════════════════════════════════════════════════════
# TRAINING LOOP
# ═══════════════════════════════════════════════════════════════════════════════

def train_gnn(
    feature_matrix:  pd.DataFrame,
    df_transactions: pd.DataFrame,
    feat_cols:       list[str],
    model_type:      str = "sage",     # "sage" | "gat"
    n_epochs:        int = 200,
    lr:              float = 0.005,
    patience:        int = 25,
    hidden1:         int = 128,
    hidden2:         int = 64,
) -> dict:
    """
    Train a GNN model on the transaction graph.

    Parameters
    ----------
    feature_matrix  : master feature DataFrame (from feature_engineering.py)
    df_transactions : canonical transaction DataFrame
    feat_cols       : list of feature column names to use
    model_type      : "sage" for GraphSAGE, "gat" for GAT
    n_epochs        : maximum training epochs
    lr              : learning rate
    patience        : early stopping patience (epochs without improvement)
    hidden1/hidden2 : hidden layer dimensions

    Returns
    -------
    dict with keys: model, scaler, account_idx, data, results, proba_series
    """
    if not TORCH_AVAILABLE:
        print("\n  [GNN] torch/torch_geometric not installed — skipping GNN.")
        print("  Install with: pip install torch torch_geometric")
        return {}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'─'*55}")
    print(f"  Training {model_type.upper()} GNN on {device}")
    print(f"{'─'*55}")

    # ── Prepare data ──────────────────────────────────────────────────────────
    data, scaler, account_idx = prepare_pyg_data(
        feature_matrix, df_transactions, feat_cols
    )
    data = data.to(device)
    in_channels = data.num_node_features

    # ── Class weights for imbalance ────────────────────────────────────────────
    n_mule   = int(data.y.sum().item())
    n_normal = data.num_nodes - n_mule
    w_mule   = n_normal / max(n_mule, 1)
    class_w  = torch.tensor([1.0, w_mule], dtype=torch.float).to(device)
    print(f"  Class weight for mule: {w_mule:.1f}×")

    # ── Model ─────────────────────────────────────────────────────────────────
    torch.manual_seed(RANDOM_STATE)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_STATE)
        
    if model_type == "gat":
        model = GATDetector(in_channels, hidden=hidden2).to(device)
    else:
        model = GraphSAGEDetector(in_channels, hidden1=hidden1,
                                   hidden2=hidden2).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model params: {n_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=10, factor=0.5, min_lr=1e-5
    )

    # ── Training ──────────────────────────────────────────────────────────────
    def train_step():
        model.train()
        optimizer.zero_grad()
        logits, _ = model(data.x, data.edge_index)
        loss = F.cross_entropy(
            logits[data.train_mask],
            data.y[data.train_mask],
            weight=class_w,
        )
        loss.backward()
        optimizer.step()
        return loss.item()

    @torch.no_grad()
    def evaluate(mask):
        model.eval()
        logits, _ = model(data.x, data.edge_index)
        proba     = F.softmax(logits, dim=1)[:, 1]
        preds     = logits.argmax(dim=1)
        y_true    = data.y[mask].cpu().numpy()
        y_prob    = proba[mask].cpu().numpy()
        y_pred    = preds[mask].cpu().numpy()
        ap  = average_precision_score(y_true, y_prob) if y_true.sum() > 0 else 0.0
        roc = roc_auc_score(y_true, y_prob)          if y_true.sum() > 0 else 0.0
        return ap, roc, y_true, y_prob, y_pred

    best_val_ap = 0.0
    best_state  = None
    no_improve  = 0
    history     = {"loss": [], "val_ap": [], "val_roc": []}

    print(f"\n  Epoch  Loss     Val-PR-AUC  Val-ROC-AUC")
    print(f"  {'─'*45}")

    for epoch in range(1, n_epochs + 1):
        loss = train_step()
        history["loss"].append(loss)

        if epoch % 5 == 0:
            val_ap, val_roc, *_ = evaluate(data.val_mask)
            history["val_ap"].append(val_ap)
            history["val_roc"].append(val_roc)
            scheduler.step(val_ap)

            marker = ""
            if val_ap > best_val_ap:
                best_val_ap = val_ap
                best_state  = {k: v.clone() for k, v in model.state_dict().items()}
                no_improve  = 0
                marker      = " ← best"
            else:
                no_improve += 1

            if epoch % 20 == 0 or marker:
                print(f"  {epoch:4d}   {loss:.4f}   {val_ap:.4f}       "
                      f"{val_roc:.4f}{marker}")

            if no_improve >= patience // 5:
                print(f"\n  Early stopping at epoch {epoch} "
                      f"(best val PR-AUC = {best_val_ap:.4f})")
                break

    # ── Test evaluation ────────────────────────────────────────────────────────
    if best_state is not None:
        model.load_state_dict(best_state)

    test_ap, test_roc, y_te, y_prob_te, y_pred_te = evaluate(data.test_mask)

    print(f"\n  ── Test Results ({model_type.upper()}) ──────────────────")
    print(f"  PR-AUC  : {test_ap:.4f}")
    print(f"  ROC-AUC : {test_roc:.4f}")
    print(classification_report(y_te, y_pred_te,
                                target_names=["Normal", "Mule"], digits=4))

    # ── Full-graph probabilities ───────────────────────────────────────────────
    model.eval()
    with torch.no_grad():
        logits, embeddings = model(data.x, data.edge_index)
        all_proba = F.softmax(logits, dim=1)[:, 1].cpu().numpy()
        all_emb   = embeddings.cpu().numpy()

    # Map back to account names
    idx_to_acc   = {v: k for k, v in account_idx.items()}
    accounts_ord = [idx_to_acc[i] for i in range(len(all_proba))]
    proba_series = pd.Series(all_proba, index=accounts_ord, name="gnn_proba")
    emb_df       = pd.DataFrame(all_emb, index=accounts_ord)

    # ── Save ──────────────────────────────────────────────────────────────────
    model_path = OUTPUTS_MODELS / f"gnn_{model_type}.pt"
    torch.save({
        "model_state":   best_state or model.state_dict(),
        "model_type":    model_type,
        "in_channels":   in_channels,
        "hidden1":       hidden1,
        "hidden2":       hidden2,
        "feat_cols":     feat_cols,
        "account_idx":   account_idx,
    }, model_path)
    print(f"\n  Saved model → {model_path}")

    proba_series.to_csv(OUTPUTS_RESULTS / f"gnn_{model_type}_proba.csv", header=True)

    return {
        "model":       model,
        "scaler":      scaler,
        "account_idx": account_idx,
        "data":        data,
        "proba_series": proba_series,
        "embeddings":   emb_df,
        "history":      history,
        "test_results": {
            "Model":    f"GNN-{model_type.upper()}",
            "Precision": float(np.mean(y_pred_te[y_te == 1] == 1)) if y_te.sum() > 0 else 0,
            "Recall":    float(y_pred_te[y_te == 1].mean()) if y_te.sum() > 0 else 0,
            "PR-AUC":   float(test_ap),
            "ROC-AUC":  float(test_roc),
            "y_test":      y_te,
            "y_prob_test": y_prob_te,
            "y_pred_test": y_pred_te,
        },
    }


def train_graph_sage(
    feature_matrix: pd.DataFrame,
    df_transactions: pd.DataFrame,
    feat_cols: list[str],
    **kwargs,
) -> dict:
    """Convenience wrapper for the default GraphSAGE detector."""
    return train_gnn(
        feature_matrix=feature_matrix,
        df_transactions=df_transactions,
        feat_cols=feat_cols,
        model_type="sage",
        **kwargs,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# EMBEDDING VISUALISATION (t-SNE)
# ═══════════════════════════════════════════════════════════════════════════════

def plot_embeddings(
    embeddings:     pd.DataFrame,
    feature_matrix: pd.DataFrame,
    save:           bool = True,
):
    """
    Reduce GNN node embeddings to 2D with t-SNE and plot.
    Mule accounts should cluster visibly away from normal accounts.
    """
    try:
        from sklearn.manifold import TSNE
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib or sklearn not available for embedding plot")
        return

    print("\n  Plotting GNN node embeddings (t-SNE)...")

    fm = feature_matrix.set_index("account") if "account" in feature_matrix.columns \
         else feature_matrix

    common = embeddings.index.intersection(fm.index)
    X      = embeddings.loc[common].values
    labels = fm.loc[common, "is_fraud"].fillna(0).astype(int).values

    # Limit to 3000 points for speed
    if len(X) > 3000:
        rng  = np.random.default_rng(RANDOM_STATE)
        idx  = rng.choice(len(X), 3000, replace=False)
        X      = X[idx]
        labels = labels[idx]

    # sklearn changed TSNE arg name from n_iter -> max_iter in newer releases.
    tsne_kwargs = {
        "n_components": 2,
        "perplexity": 30,
        "random_state": RANDOM_STATE,
    }
    if "max_iter" in TSNE.__init__.__code__.co_varnames:
        tsne_kwargs["max_iter"] = 500
    else:
        tsne_kwargs["n_iter"] = 500

    tsne = TSNE(**tsne_kwargs)
    X_2d  = tsne.fit_transform(X)

    fig, ax = plt.subplots(figsize=(9, 7))
    colors = [NODE_COLOR_MULE if l == 1 else NODE_COLOR_NORMAL for l in labels]
    sizes  = [60 if l == 1 else 15 for l in labels]
    ax.scatter(X_2d[:, 0], X_2d[:, 1], c=colors, s=sizes, alpha=0.7, linewidths=0)

    import matplotlib.patches as mpatches
    legend = [
        mpatches.Patch(color=NODE_COLOR_MULE,   label=f"Mule ({labels.sum():,})"),
        mpatches.Patch(color=NODE_COLOR_NORMAL, label=f"Normal ({(labels==0).sum():,})"),
    ]
    ax.legend(handles=legend, frameon=False, fontsize=11)
    ax.set_title("GNN Node Embeddings — t-SNE Projection", fontsize=13, fontweight="bold")
    ax.set_xlabel("t-SNE dim 1"); ax.set_ylabel("t-SNE dim 2")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()

    if save:
        path = OUTPUTS_PLOTS / "10_gnn_embeddings_tsne.png"
        fig.savefig(path, dpi=PLOT_DPI, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved → {path}")

    return fig
