# scrapers/scrape_basic_logs_fast.py

import sys, os, time, requests, pandas as pd

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
sys.path.insert(0, ROOT_DIR)

# REAL NBA HEADERS - REQUIRED
HEADERS = {
    "Host": "stats.nba.com",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
    "Accept": "application/json, text/plain, */*",
    "Referer": "https://www.nba.com/",
    "Origin": "https://www.nba.com",
    "Connection": "keep-alive",
}

def scrape_fast(season="2024-25"):
    """
    Scrape NBA game logs reliably from stats.nba.com 
    using the leaguegamelog endpoint.
    """

    print("🔍 Fetching NBA Game Logs (Official Stats API)...")

    url = "https://stats.nba.com/stats/leaguegamelog"
    params = {
        "Counter": "10000",
        "Direction": "DESC",
        "LeagueID": "00",
        "PlayerOrTeam": "P",
        "Season": season,
        "SeasonType": "Regular Season",
        "Sorter": "PLAYER_NAME"
    }

    r = requests.get(url, headers=HEADERS, params=params)

    if r.status_code != 200:
        raise Exception(f"NBA API error: {r.status_code}")

    data = r.json()

    rows = data["resultSets"][0]["rowSet"]
    headers = data["resultSets"][0]["headers"]

    df = pd.DataFrame(rows, columns=headers)

    # Standardize column names
    df = df.rename(columns={
        "PLAYER_NAME": "player_name",
        "PTS": "points",
        "REB": "rebounds",
        "AST": "assists",
        "MIN": "minutes",
        "GAME_DATE": "GAME_DATE",
    })

    # Force minutes into number
    df["minutes"] = pd.to_numeric(df["minutes"], errors="coerce")

    # SAVE FILE
    os.makedirs("data", exist_ok=True)
    df.to_csv("data/player_game_logs_raw.csv", index=False)

    print(f"Saved RAW logs → data/player_game_logs_raw.csv")
    print(f"Rows: {len(df)}")
    return df


if __name__ == "__main__":
    scrape_fast()
