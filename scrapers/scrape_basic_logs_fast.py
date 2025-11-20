import sys, os, pandas as pd, requests

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
sys.path.insert(0, ROOT_DIR)


def scrape_fast(season="2025-26"):
    """
    Scrape FULL NBA game logs for any season using the NBA Stats Proxy.
    This bypasses Cloudflare and works perfectly in Streamlit Cloud.
    """

    print(f"🔍 Fetching NBA Game Logs for {season} (NBA Stats Proxy)...")

    url = "https://nba-stats-proxy.fly.dev/leaguegamelog"
    params = {
        "Season": season,
        "SeasonType": "Regular Season",
        "PlayerOrTeam": "P"
    }

    r = requests.get(url, params=params, timeout=25)
    if r.status_code != 200:
        raise Exception(f"API error {r.status_code}: {r.text}")

    data = r.json()

    # Standard NBA Stats API structure
    result = data["resultSets"][0]
    headers = result["headers"]
    rows = result["rowSet"]

    if len(rows) == 0:
        raise Exception("❌ No logs returned — season may not be supported yet.")

    df = pd.DataFrame(rows, columns=headers)

    # Normalize useful columns
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