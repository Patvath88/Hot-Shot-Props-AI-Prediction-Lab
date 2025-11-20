import pandas as pd
import os

def build_dataset():
    raw = "data/player_game_logs_raw.csv"

    if not os.path.exists(raw):
        raise Exception("❌ No raw logs found. Run scraper first.")

    df = pd.read_csv(raw)

    df = df.sort_values(["player_name", "GAME_DATE"])

    # Rolling windows
    df["points_rolling_5"] = df.groupby("player_name")["points"].rolling(5).mean().reset_index(0, drop=True)
    df["rebounds_rolling_5"] = df.groupby("player_name")["rebounds"].rolling(5).mean().reset_index(0, drop=True)
    df["assists_rolling_5"] = df.groupby("player_name")["assists"].rolling(5).mean().reset_index(0, drop=True)

    df["minutes_rolling_5"] = df.groupby("player_name")["minutes"].rolling(5).mean().reset_index(0, drop=True)

    df["form_score"] = (
        df["points_rolling_5"] * 0.5 +
        df["rebounds_rolling_5"] * 0.2 +
        df["assists_rolling_5"] * 0.3
    )

    df = df.dropna()

    out_path = "data/player_game_logs.csv"
    df.to_csv(out_path, index=False)

    print(f"✅ Dataset built → {out_path} | Rows: {len(df)}")

    return df


if __name__ == "__main__":
    build_dataset()