# scrapers/scrape_basic_logs_fast.py

import sys, os, pandas as pd, requests

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
sys.path.insert(0, ROOT_DIR)


def scrape_fast(season="2024-25"):
    """
    CLOUD-SAFE SCRAPER:
    Uses a stable NBA stats mirror that works on Streamlit Cloud.
    """

    print("🔍 Fetching NBA game logs (Cloud-Safe Mirror)...")

    # This endpoint mirrors the official gamelog API
    url = "https://nba-api.fly.dev/leaguegamelog"

    params = {
        "season": season,
        "seasonType": "Regular Season"
    }

    response = requests.get(url, params=params, timeout=15)

    if response.status_code != 200:
        raise Exception(f"API error: {response.status_code}\n{response.text}")

    data = response.json()

    if "rowSet" not in data:
        raise Exception("Invalid response format from API mirror.")

    rows = data["rowSet"]
    headers = data["headers"]

    df = pd.DataFrame(rows, columns=headers)

    # Standardize columns
    df = df.rename(columns={
        "PLAYER_NAME": "player_name",
        "PTS": "points",
        "REB": "rebounds",
        "AST": "assists",
        "MIN": "minutes",
        "GAME_DATE": "GAME_DATE"
    })

    df["minutes"] = pd.to_numeric(df["minutes"], errors="coerce")
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"], errors="coerce")

    os.makedirs("data", exist_ok=True)
    df.to_csv("data/player_game_logs_raw.csv", index=False)

    print(f"Saved RAW logs → data/player_game_logs_raw.csv")
    print(f"Rows: {len(df)}")

    return df


# IMPORTANT: ❗ REMOVE AUTO-RUNNING
# DO NOT EXECUTE scraper on import.
# Streamlit Cloud imports all files on startup.
#
# if __name__ == "__main__":
#     scrape_fast()