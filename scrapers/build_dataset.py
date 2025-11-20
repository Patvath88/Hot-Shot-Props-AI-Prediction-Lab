import pandas as pd
import os

def build_dataset():
    print("🔨 Building ML Dataset...")

    df = pd.read_csv("data/player_game_logs_raw.csv")

    df = df.sort_values(["player_id", "GAME_DATE"])

    df["points_rolling_5"] = df.groupby("player_id")["points"].rolling(5).mean().reset_index(0, drop=True)
    df["rebounds_rolling_5"] = df.groupby("player_id")["rebounds"].rolling(5).mean().reset_index(0, drop=True)
    df["assists_rolling_5"] = df.groupby("player_id")["assists"].rolling(5).mean().reset_index(0, drop=True)
    df["minutes_rolling"] = df.groupby("player_id")["minutes"].rolling(5).mean().reset_index(0, drop=True)

    df = df.dropna()

    df.to_csv("data/player_game_logs.csv", index=False)
    print("Saved → data/player_game_logs.csv")
    print("Rows:", len(df))

    return df


if __name__ == "__main__":
    build_dataset()