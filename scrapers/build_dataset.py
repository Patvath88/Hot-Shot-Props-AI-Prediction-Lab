# scrapers/build_dataset.py

import sys, os
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
sys.path.insert(0, ROOT_DIR)

import pandas as pd
from utils.features import add_basic_features


def build_dataset():
    print("📦 Loading raw logs...")

    raw_path = "data/player_game_logs_raw.csv"
    if not os.path.exists(raw_path):
        raise FileNotFoundError("❌ Raw logs missing — run scraper first!")

    df = pd.read_csv(raw_path)

    if "GAME_DATE" not in df.columns:
        raise ValueError("❌ GAME_DATE not found in raw logs.")

    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"], errors="coerce")

    required = ["player_name", "points", "rebounds", "assists", "minutes"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"❌ Missing base column: {col}")

    df = add_basic_features(df)

    print("📊 Final dataset columns:", list(df.columns))
    print("📈 Rows:", len(df))

    os.makedirs("data", exist_ok=True)
    df.to_csv("data/player_game_logs.csv", index=False)

    print("✅ Saved processed dataset → data/player_game_logs.csv")


# IMPORTANT: Do NOT auto-run.
# Streamlit imports this file on startup, so auto-running breaks the app.
#
# if __name__ == "__main__":
#     build_dataset()