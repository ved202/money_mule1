"""
config/config.py
================
Central configuration for the entire mule detection system.
All thresholds, paths, and hyperparameters live here so that
nothing is hardcoded inside business logic modules.
"""

import os
from pathlib import Path

# -- Root paths ----------------------------------------------------------------
ROOT_DIR        = Path(__file__).resolve().parent.parent

# Allow overriding the data folder via environment variable
DATA_BASE       = os.getenv("MULE_DATA_DIR", "data")
DATA_RAW        = ROOT_DIR / DATA_BASE / "raw"
DATA_PROCESSED  = ROOT_DIR / DATA_BASE / "processed"
DATA_SYNTHETIC  = ROOT_DIR / DATA_BASE / "synthetic"
# Allow overriding the output folder via environment variable
OUTPUT_BASE     = os.getenv("MULE_OUTPUT_DIR", "outputs")
OUTPUTS_PLOTS   = ROOT_DIR / OUTPUT_BASE / "plots"
OUTPUTS_REPORTS = ROOT_DIR / OUTPUT_BASE / "reports"
OUTPUTS_MODELS  = ROOT_DIR / OUTPUT_BASE / "models"
OUTPUTS_RESULTS = ROOT_DIR / OUTPUT_BASE / "results"

for d in [DATA_RAW, DATA_PROCESSED, DATA_SYNTHETIC,
          OUTPUTS_PLOTS, OUTPUTS_REPORTS, OUTPUTS_MODELS, OUTPUTS_RESULTS]:
    d.mkdir(parents=True, exist_ok=True)

# -- Dataset settings ----------------------------------------------------------
PAYSIM_PATH  = DATA_RAW / "paysim.csv"
AMLSIM_PATH  = DATA_RAW / "amlsim.csv"
SYNTHETIC_TX = DATA_SYNTHETIC / "synthetic_transactions.csv"

# Columns expected in transaction data (canonical schema)
TX_SCHEMA = {
    "sender":    "sender_account",
    "receiver":  "receiver_account",
    "amount":    "transaction_amount",
    "timestamp": "timestamp",
    "label":     "is_fraud",      # optional; may not exist in raw data
}

# -- Data generation (synthetic dataset) --------------------------------------
SYNTH_N_ACCOUNTS       = 5_000     # total accounts in synthetic dataset
SYNTH_N_TRANSACTIONS   = 50_000    # total transactions
SYNTH_MULE_FRACTION    = 0.04      # 4% of accounts are mules
SYNTH_N_MULE_NETWORKS  = 15        # number of coordinated mule networks
SYNTH_NETWORK_SIZE_MIN = 5         # min accounts per mule network
SYNTH_NETWORK_SIZE_MAX = 25        # max accounts per mule network
SYNTH_SEED             = 42

# -- Graph construction --------------------------------------------------------
GRAPH_MIN_EDGE_WEIGHT  = 0         # minimum transaction count to keep an edge
GRAPH_SELF_LOOPS       = False     # whether to allow self-loops

# -- Feature engineering -------------------------------------------------------
BETWEENNESS_K          = 200       # number of samples for approx. betweenness
PAGERANK_ALPHA         = 0.85
PAGERANK_MAX_ITER      = 200
LARGE_GRAPH_THRESHOLD  = 20_000    # reduce expensive graph metrics beyond this
LARGE_BETWEENNESS_K    = 64
CLUSTERING_SKIP_THRESHOLD = 50_000 # skip clustering on very large graphs

# -- Lifecycle detection -------------------------------------------------------
WINDOW_SIZES_DAYS      = [1, 3, 7, 14, 30]   # sliding windows for temporal features
TEST_TX_MAX_AMOUNT     = 500        # amounts below this = potential test transaction
TEST_TX_ROUND_MODULO   = 100        # round amount modulo threshold
DORMANT_GAP_DAYS       = 14         # silence before "sudden wakeup"
BURST_RATIO_THRESHOLD  = 4.0        # peak / avg weekly txns = burst

# -- Community detection -------------------------------------------------------
LOUVAIN_RESOLUTION     = 0.1        # Lower = larger communities
MIN_COMMUNITY_SIZE     = 3          # ignore singleton/pair communities
SUSPICIOUS_COMMUNITY_MULE_RATE = 0.30  # community with >30% mules is flagged

# -- Model hyperparameters -----------------------------------------------------
RF_N_ESTIMATORS        = 300
RF_MAX_DEPTH           = None
RF_MIN_SAMPLES_LEAF    = 2
RF_CLASS_WEIGHT        = "balanced"

GB_N_ESTIMATORS        = 200
GB_LEARNING_RATE       = 0.05
GB_MAX_DEPTH           = 5

ISO_N_ESTIMATORS       = 200
ISO_CONTAMINATION      = "auto"     # set to float if mule rate is known

# -- GNN / Graph DB extensions ------------------------------------------------
GNN_DEFAULT_MODEL      = "sage"
GNN_DEFAULT_EPOCHS     = 200
GNN_DEFAULT_PATIENCE   = 25
GNN_HIDDEN1            = 128
GNN_HIDDEN2            = 64

NEO4J_URI              = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER             = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD         = os.getenv("NEO4J_PASSWORD", "")
NEO4J_DATABASE         = os.getenv("NEO4J_DATABASE", "neo4j")
NEO4J_BATCH_SIZE       = 5_000

# -- Evaluation ----------------------------------------------------------------
TEST_SIZE              = 0.20
RANDOM_STATE           = 42
CV_FOLDS               = 5
VALIDATION_SIZE        = 0.20

# -- Visualization -------------------------------------------------------------
PLOT_DPI               = 150
PLOT_FIGSIZE_LARGE     = (14, 10)
PLOT_FIGSIZE_MEDIUM    = (10, 7)
PLOT_FIGSIZE_SMALL     = (8, 5)
NODE_COLOR_NORMAL      = "#85B7EB"
NODE_COLOR_MULE        = "#E24B4A"
NODE_COLOR_SUSPECTED   = "#EF9F27"
EDGE_COLOR_NORMAL      = "#B4B2A9"
EDGE_COLOR_SUSPICIOUS  = "#D85A30"
