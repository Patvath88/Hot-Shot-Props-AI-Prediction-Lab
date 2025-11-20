# scrapers/build_dataset.py

import sys, os
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
sys.path.insert(0, ROOT_DIR)

import pandas as pd
from utils.features import add_basic_features


def build_dataset():
    print("📦 Building dataset...")

    raw = "data/player_game_logs_raw.csv"
    if not os.path.exists(raw):
        raise FileNotFoundError("❌ Raw logs missing — run scraper first!")

    df = pd.read_csv(raw)
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])

    base_cols = ["player_name", "points", "rebounds", "assists", "minutes"]
    for c in base_cols:
        if c not in df.columns:
            raise ValueError(f"❌ Missing column: {c}")

    df = add_basic_features(df)

    os.makedirs("data", exist_ok=True)
    df.to_csv("data/player_game_logs.csv", index=False)

    print("✅ Dataset saved → data/player_game_logs.csv")