# scrapers/build_dataset.py

import pandas as pd
from utils.features import add_basic_features


def build_dataset():
    print("Loading raw logs...")
    df = pd.read_csv("data/player_game_logs_raw.csv")

    # Ensure GAME_DATE is a date
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])

    # Add derived features
    df = add_basic_features(df)

    # Keep only necessary columns
    needed = [
        "player_name", "GAME_DATE",
        "points", "rebounds", "assists", "minutes",
    ]
    df = df[[c for c in df.columns if c in needed or "rolling" in c or "avg_season" in c or c == "form_score"]]

    df.to_csv("data/player_game_logs.csv", index=False)

    print("Saved → data/player_game_logs.csv")
    print("Rows:", len(df))

    return df


if __name__ == "__main__":
    build_dataset()
