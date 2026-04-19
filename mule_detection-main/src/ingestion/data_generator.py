"""
src/ingestion/data_generator.py
================================
Generates a realistic synthetic AML transaction dataset that simulates
PaySim / AMLSim style data.

Design principles (matching real mule network behaviour):
  - Normal accounts: moderate, consistent transaction patterns
  - Mule networks:   coordinated fan-in → pass-through → fan-out patterns
  - Layering:        mule accounts receive from "dirty" sources then split
                     the funds across many accounts to obscure origin

Output schema (canonical):
  sender_account | receiver_account | transaction_amount | timestamp | is_fraud
"""

import numpy as np
import pandas as pd
import random
from datetime import datetime, timedelta
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from config.config import *

rng = np.random.default_rng(SYNTH_SEED)
random.seed(SYNTH_SEED)


def _account_id(i: int) -> str:
    return f"ACC{i:06d}"


def _random_amount(low=10, high=50_000, skew=True) -> float:
    """Log-normal distribution to mimic real transaction amounts."""
    if skew:
        val = rng.lognormal(mean=6.5, sigma=1.8)
        return round(float(np.clip(val, low, high)), 2)
    return round(rng.uniform(low, high), 2)


def _round_amount(approx: float) -> float:
    """Snap to a round number — used for synthetic test transactions."""
    for base in [1000, 500, 100, 50]:
        if approx > base:
            return float(round(approx / base) * base)
    return float(round(approx / 10) * 10)


def generate_synthetic_dataset(
    n_accounts: int = SYNTH_N_ACCOUNTS,
    n_transactions: int = SYNTH_N_TRANSACTIONS,
    mule_fraction: float = SYNTH_MULE_FRACTION,
    n_mule_networks: int = SYNTH_N_MULE_NETWORKS,
    save: bool = True,
) -> pd.DataFrame:
    """
    Build a synthetic transaction dataset with embedded mule networks.

    Mule network structure (3-layer model):
        Layer 0: Fraud source accounts  (inject dirty money)
        Layer 1: Mule accounts          (receive and hold briefly)
        Layer 2: Exit accounts          (cash-out destinations)

    Returns DataFrame with canonical schema.
    """
    print("=" * 60)
    print("  Generating synthetic AML transaction dataset")
    print("=" * 60)

    # ── Account pool ──────────────────────────────────────────────────────────
    all_ids = list(range(n_accounts))
    n_mules = int(n_accounts * mule_fraction)

    # Assign mule accounts to networks
    mule_ids:   list[int] = []
    mule_label: dict[int, int] = {}   # account_id → 1 if mule
    mule_net:   dict[int, int] = {}   # account_id → network_id

    net_id = 0
    while len(mule_ids) < n_mules and net_id < n_mule_networks:
        size = random.randint(SYNTH_NETWORK_SIZE_MIN, SYNTH_NETWORK_SIZE_MAX)
        candidates = [i for i in all_ids if i not in mule_label]
        if len(candidates) < size:
            break
        net_members = random.sample(candidates, size)
        for m in net_members:
            mule_ids.append(m)
            mule_label[m] = 1
            mule_net[m]   = net_id
        net_id += 1

    normal_ids = [i for i in all_ids if i not in mule_label]
    print(f"  Accounts      : {n_accounts:,}")
    print(f"  Mule accounts : {len(mule_ids):,} across {net_id} networks")
    print(f"  Normal accts  : {len(normal_ids):,}")

    # ── Timeline ──────────────────────────────────────────────────────────────
    start_dt = datetime(2023, 1, 1)
    end_dt   = datetime(2023, 6, 30)
    span_sec = int((end_dt - start_dt).total_seconds())

    records = []

    # ── Normal transactions ───────────────────────────────────────────────────
    n_normal_tx = int(n_transactions * 0.75)
    print(f"  Generating {n_normal_tx:,} normal transactions...")
    for _ in range(n_normal_tx):
        src, dst = random.sample(normal_ids + mule_ids, 2)
        ts = start_dt + timedelta(seconds=int(rng.uniform(0, span_sec)))
        records.append({
            "sender_account":    _account_id(src),
            "receiver_account":  _account_id(dst),
            "transaction_amount": _random_amount(),
            "timestamp":         ts,
            "is_fraud":          mule_label.get(src, 0),
        })

    # ── Mule network transactions ─────────────────────────────────────────────
    # Pattern: source → mule layer1 → mule layer2 → exit
    n_mule_tx = n_transactions - n_normal_tx
    print(f"  Generating {n_mule_tx:,} mule-network transactions...")

    for net in range(net_id):
        net_members = [m for m in mule_ids if mule_net[m] == net]
        if len(net_members) < 2:
            continue

        # Split into collector (receives dirty money) and distributor (sends out)
        mid = len(net_members) // 2
        collectors    = net_members[:mid]   or net_members[:1]
        distributors  = net_members[mid:]   or net_members[-1:]

        # Fraud source accounts (outside the mule network, inject dirty funds)
        sources = random.sample(normal_ids, min(3, len(normal_ids)))
        exits   = random.sample([i for i in normal_ids
                                  if i not in sources], min(5, len(normal_ids) - 3))

        net_tx_count = max(20, n_mule_tx // net_id)

        # Phase 1 — Activation: small test transactions
        n_test = random.randint(2, 5)
        test_start = start_dt + timedelta(days=random.randint(0, 30))
        for _ in range(n_test):
            src = random.choice(sources)
            dst = random.choice(collectors)
            amt = _round_amount(random.uniform(50, 400))
            ts  = test_start + timedelta(hours=random.randint(0, 48))
            records.append({
                "sender_account":    _account_id(src),
                "receiver_account":  _account_id(dst),
                "transaction_amount": amt,
                "timestamp":         ts,
                "is_fraud":          1,
            })

        # Phase 2 — Laundering: large rapid transfers into mule accounts
        launder_start = test_start + timedelta(days=random.randint(5, 20))
        n_launder = int(net_tx_count * 0.4)
        for _ in range(n_launder):
            src = random.choice(sources)
            dst = random.choice(collectors)
            amt = _random_amount(5_000, 50_000, skew=False)
            ts  = launder_start + timedelta(
                hours=rng.integers(0, 72).item())
            records.append({
                "sender_account":    _account_id(src),
                "receiver_account":  _account_id(dst),
                "transaction_amount": amt,
                "timestamp":         ts,
                "is_fraud":          1,
            })

        # Phase 3 — Layering: pass money between mule accounts
        n_layer = int(net_tx_count * 0.35)
        for _ in range(n_layer):
            src = random.choice(collectors)
            dst = random.choice(distributors)
            amt = _random_amount(1_000, 20_000, skew=False)
            ts  = launder_start + timedelta(
                hours=rng.integers(24, 120).item())
            records.append({
                "sender_account":    _account_id(src),
                "receiver_account":  _account_id(dst),
                "transaction_amount": amt,
                "timestamp":         ts,
                "is_fraud":          1,
            })

        # Phase 4 — Exit: distribute to exit accounts (cash-out)
        n_exit = int(net_tx_count * 0.25)
        for _ in range(n_exit):
            src = random.choice(distributors)
            dst = random.choice(exits)
            amt = _random_amount(500, 10_000, skew=False)
            ts  = launder_start + timedelta(
                hours=rng.integers(72, 200).item())
            records.append({
                "sender_account":    _account_id(src),
                "receiver_account":  _account_id(dst),
                "transaction_amount": amt,
                "timestamp":         ts,
                "is_fraud":          1,
            })

    # ── Assemble DataFrame ────────────────────────────────────────────────────
    df = pd.DataFrame(records)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["tx_id"] = [f"TX{i:08d}" for i in range(len(df))]

    fraud_rate = df["is_fraud"].mean()
    print(f"\n  Total transactions : {len(df):,}")
    print(f"  Fraud transactions : {df['is_fraud'].sum():,} ({fraud_rate:.2%})")
    print(f"  Date range         : {df['timestamp'].min().date()} → "
          f"{df['timestamp'].max().date()}")

    if save:
        path = SYNTHETIC_TX
        df.to_csv(path, index=False)
        print(f"\n  Saved → {path}")

    return df


if __name__ == "__main__":
    df = generate_synthetic_dataset()
    print("\nSample rows:")
    print(df.head(5).to_string())
