import sys, os
import pandas as pd
from config.config import *
from src.visualization.visualizer import plot_lifecycle_distribution

df = pd.read_csv(DATA_PROCESSED / "lifecycle_results.csv")
plot_lifecycle_distribution(df, save=True)
print("Lifecycle plot updated.")
