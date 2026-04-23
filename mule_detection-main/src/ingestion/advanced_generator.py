import numpy as np
import pandas as pd
import random
from datetime import datetime, timedelta
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from config.config import SYNTH_SEED, DATA_SYNTHETIC

rng = np.random.default_rng(SYNTH_SEED)
random.seed(SYNTH_SEED)

def _account_id(prefix: str, i: int) -> str:
    return f"{prefix}{i:06d}"

def _random_amount(low=10, high=50_000, skew=True) -> float:
    if skew:
        val = rng.lognormal(mean=6.5, sigma=1.8)
        return round(float(np.clip(val, low, high)), 2)
    return round(rng.uniform(low, high), 2)

def generate_advanced_synthetic_dataset(
    n_accounts: int = 5000,
    save: bool = True,
) -> pd.DataFrame:
    print("=" * 60)
    print("  Generating Advanced Synthetic AML Dataset")
    print("  (60% Mules with strict Lifecycle Behaviors)")
    print("=" * 60)

    n_mules = int(n_accounts * 0.6)
    n_normal = n_accounts - n_mules
    
    # 5 Lifecycle stages
    stage_size = n_mules // 5
    mules_dormant     = [_account_id("M_DOR_", i) for i in range(stage_size)]
    mules_recruitment = [_account_id("M_REC_", i) for i in range(stage_size)]
    mules_activation  = [_account_id("M_ACT_", i) for i in range(stage_size)]
    mules_laundering  = [_account_id("M_LAU_", i) for i in range(stage_size)]
    mules_exit        = [_account_id("M_EXT_", i) for i in range(stage_size)]
    
    all_mules = mules_dormant + mules_recruitment + mules_activation + mules_laundering + mules_exit
    normal_accounts = [_account_id("N_", i) for i in range(n_normal)]
    
    print(f"  Total Accounts: {n_accounts:,}")
    print(f"  Normal: {len(normal_accounts):,}")
    print(f"  Mules : {len(all_mules):,} ({len(mules_dormant)} per lifecycle stage)")

    start_dt = datetime(2023, 1, 1)
    end_dt   = datetime(2023, 6, 30)
    span_days = (end_dt - start_dt).days

    records = []

    # Helper to create transaction
    def add_tx(src, dst, amt, ts, is_fraud=0):
        records.append({
            "sender_account": src,
            "receiver_account": dst,
            "transaction_amount": round(amt, 2),
            "timestamp": ts,
            "is_fraud": is_fraud
        })

    # 1. Normal Accounts (Background noise)
    print("  Generating normal transactions...")
    for _ in range(n_normal * 10):
        src = random.choice(normal_accounts)
        dst = random.choice(normal_accounts)
        ts = start_dt + timedelta(days=random.randint(0, span_days), hours=random.randint(0, 23))
        add_tx(src, dst, _random_amount(10, 5000), ts, 0)

    # Dedicated pools so we don't accidentally taint all normal accounts
    victims = normal_accounts[:50]
    exits = normal_accounts[-50:]

    # 2. Dormant Mules
    print("  Generating Dormant mules...")
    for mule in mules_dormant:
        n_tx = random.randint(1, 2)
        for _ in range(n_tx):
            dst = random.choice(exits)
            ts = start_dt + timedelta(days=random.randint(0, 10), hours=random.randint(0, 23))
            add_tx(mule, dst, _random_amount(50, 1000), ts, 1)

    # 3. Recruitment Mules
    print("  Generating Recruitment mules...")
    for mule in mules_recruitment:
        n_deposits = random.randint(5, 15)
        for _ in range(n_deposits):
            src = random.choice(victims)
            ts = start_dt + timedelta(days=random.randint(0, span_days), hours=random.randint(0, 23))
            add_tx(src, mule, random.uniform(10, 200), ts, 1)

    # 4. Activation Mules
    print("  Generating Activation mules...")
    for mule in mules_activation:
        window_start = start_dt + timedelta(days=random.randint(20, span_days - 20))
        n_test = random.randint(4, 10)
        round_amounts = [50.0, 100.0, 200.0, 500.0]
        for _ in range(n_test):
            src = random.choice(victims)
            ts = window_start + timedelta(hours=random.randint(0, 72))
            amt = random.choice(round_amounts)
            add_tx(src, mule, amt, ts, 1)
            if random.random() > 0.5:
                dst = random.choice(exits)
                ts_out = ts + timedelta(hours=random.randint(1, 5))
                add_tx(mule, dst, amt, ts_out, 1)

    # 5. Laundering Mules
    print("  Generating Laundering mules...")
    for mule in mules_laundering:
        n_bursts = random.randint(2, 5)
        for _ in range(n_bursts):
            burst_start = start_dt + timedelta(days=random.randint(10, span_days - 10))
            src = random.choice(victims)
            amt = random.uniform(10000, 50000)
            add_tx(src, mule, amt, burst_start, 1)
            
            n_splits = random.randint(3, 8)
            split_amt = amt / n_splits
            for j in range(n_splits):
                dst = random.choice(exits)
                ts_out = burst_start + timedelta(hours=random.randint(1, 12))
                add_tx(mule, dst, split_amt * random.uniform(0.9, 1.0), ts_out, 1)

    # 6. Exit Mules
    print("  Generating Exit mules...")
    for mule in mules_exit:
        for _ in range(random.randint(3, 8)):
            src = random.choice(victims)
            ts = start_dt + timedelta(days=random.randint(0, span_days - 30))
            add_tx(src, mule, _random_amount(100, 2000), ts, 1)
            
        exit_day = start_dt + timedelta(days=random.randint(span_days - 20, span_days))
        drain_amt = random.uniform(20000, 80000)
        add_tx(random.choice(victims), mule, drain_amt, exit_day - timedelta(hours=24), 1)
        add_tx(mule, random.choice(exits), drain_amt, exit_day, 1)

    # Assemble and sort
    df = pd.DataFrame(records)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    
    print(f"\n  Total transactions: {len(df):,}")
    print(f"  Fraud txns: {df['is_fraud'].sum():,}")
    print(f"  Date range: {df['timestamp'].min().date()} -> {df['timestamp'].max().date()}")

    if save:
        path = DATA_SYNTHETIC / "advanced_transactions.csv"
        df.to_csv(path, index=False)
        print(f"  Saved -> {path}")

    return df

if __name__ == "__main__":
    df = generate_advanced_synthetic_dataset()
