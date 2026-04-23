import sys, os
import pandas as pd
from config.config import *
from src.visualization.visualizer import plot_temporal_drift

df = pd.read_csv(DATA_PROCESSED / "canonical_transactions.csv")
mules = df[df['is_fraud'] == 1]['sender_account'].unique()
normals = df[df['is_fraud'] == 0]['sender_account'].unique()
m_act = mules[0]
n_act = normals[0]
plot_temporal_drift(df, m_act, n_act, save=True)
print("Plot updated.")
