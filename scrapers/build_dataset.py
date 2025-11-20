# scrapers/build_dataset.py

import sys, os
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
sys.path.insert(0, ROOT_DIR)

import pandas as pd
from utils.features import add_basic_features


def build_dataset():
    print("Loading raw logs...")

    raw_path = "data/player_game_logs_raw.csv"
    if not os.path.exists(raw_path):
        raise FileNotFoundError("Raw logs missing. Run scraper first.")

    df = pd.read_csv(raw_path)

    # Ensure GAME_DATE exists and is parsed
    if "GAME_DATE" not in df.columns:
        # NBA API sometimes uses GAME_DATE column in uppercase or lowercase
        date_col = [c for c in df.columns if c.lower() == "game_date"][0]
        df.rename(columns={date_col: "GAME_DATE"}, inplace=True)

    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])

    # Must have these before engineering
    required_base = ["player_name", "points", "rebounds", "assists", "minutes"]
    for col in required_base:
        if col not in df:
            raise ValueError(f"Missing base column: {col}")

    # --- APPLY FEATURE ENGINEERING ---
    df = add_basic_features(df)

    # Print columns for debugging
    print("Final columns:", list(df.columns))
    print("Rows:", len(df))

    df.to_csv("data/player_game_logs.csv", index=False)
    print("Saved → data/player_game_logs.csv")


if __name__ == "__main__":
    build_dataset()