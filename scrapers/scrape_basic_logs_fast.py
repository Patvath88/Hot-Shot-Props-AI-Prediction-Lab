import sys, os, pandas as pd, requests

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
sys.path.insert(0, ROOT_DIR)


def scrape_fast(season="2025-26"):
    """
    Scrape FULL NBA game logs using nba-api.com mirror.
    Supports ANY season, works in Streamlit Cloud.
    """

    print(f"🔍 Fetching NBA Game Logs for {season} (nba-api.com mirror)...")

    url = "https://nba-api.com/api/leaguegamelog"
    params = {
        "Season": season,
        "SeasonType": "Regular Season",
        "PlayerOrTeam": "P"
    }

    try:
        r = requests.get(url, params=params, timeout=30)
        r.raise_for_status()
    except Exception as e:
        raise Exception(f"❌ Failed to fetch logs: {e}")

    data = r.json()

    if "resultSets" not in data:
        raise Exception("❌ Invalid response from nba-api.com")

    result = data["resultSets"][0]
    headers = result["headers"]
    rows = result["rowSet"]

    if len(rows) == 0:
        raise Exception("❌ No logs returned — season may not be published yet.")

    df = pd.DataFrame(rows, columns=headers)

    df = df.rename(columns={
        "PLAYER_NAME": "player_name",
        "PTS": "points",
        "REB": "rebounds",
        "AST": "assists",
        "MIN": "minutes",
        "GAME_DATE": "GAME_DATE",
    })

    df["minutes"] = pd.to_numeric(df["minutes"], errors="coerce")
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"], errors="coerce")

    os.makedirs(os.path.join(ROOT_DIR, "data"), exist_ok=True)
    out_path = os.path.join(ROOT_DIR, "data", "player_game_logs_raw.csv")
    df.to_csv(out_path, index=False)

    print("📁 Saved RAW logs →", out_path)
    print("📈 Total Rows:", len(df))

    return df


if __name__ == "__main__":
    scrape_fast()