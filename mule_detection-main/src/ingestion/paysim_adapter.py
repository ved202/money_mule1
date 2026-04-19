"""
src/ingestion/paysim_adapter.py
================================
Drop-in adapter for running the full pipeline on your PaySim dataset.

Your PaySim columns:
  type, amount, nameOrig, oldbalanceOrg, newbalanceOrig,
  nameDest, oldbalanceDest, newbalanceDest, isFraud, isFlaggedFraud

Usage
-----
  python src/ingestion/paysim_adapter.py --path data/raw/paysim.csv

Or import and call prepare_paysim() from main_pipeline.py via --source paysim.
"""

import pandas as pd
import numpy as np
import argparse
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from config.config import *
from src.ingestion.data_loader import load_paysim


def prepare_paysim(path: str = None, sample_n: int = None) -> pd.DataFrame:
    """
    Load PaySim CSV and enrich it with derived features before
    passing it into the canonical pipeline.

    Steps
    -----
    1. Load CSV
    2. Filter to TRANSFER and CASH_OUT (only these carry fraud labels)
    3. Reconstruct synthetic timestamps from row order (your dataset has no 'step')
    4. Derive balance-based features (drain_rate, pass-through flag)
    5. Rename to canonical schema:
         nameOrig  → sender_account
         nameDest  → receiver_account
         amount    → transaction_amount
         isFraud   → is_fraud
    6. Return canonical DataFrame ready for build_transaction_graph()
    """
    return load_paysim(path=path, sample_n=sample_n)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path",   required=True, help="Path to paysim.csv")
    parser.add_argument("--sample", type=int, default=None)
    args = parser.parse_args()

    df = prepare_paysim(path=args.path, sample_n=args.sample)
    print("\nFirst 5 rows of prepared data:")
    print(df[["sender_account","receiver_account","transaction_amount",
              "timestamp","is_fraud","sender_drained"]].head().to_string())
