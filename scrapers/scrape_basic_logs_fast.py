import sys, os, requests, pandas as pd
from datetime import datetime, timedelta

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
sys.path.insert(0, ROOT_DIR)


def fetch_scoreboard(date_str):
    """
    Fetch ESPN scoreboard for a specific YYYYMMDD string.
    Returns list of games or empty list.
    """
    url = "https://site.api.espn.com/apis/v2/sports/basketball/nba/scoreboard"
    params = {"dates": date_str}

    try:
        data = requests.get(url, params=params, timeout=20).json()
        return data.get("events", [])
    except:
        return []


def fetch_boxscore(game_id):
    """
    Fetch boxscore for a given ESPN event ID.
    Returns parsed boxscore or None.
    """
    url = f"https://site.api.espn.com/apis/v2/sports/basketball/nba/summary?event={game_id}"

    try:
        return requests.get(url, timeout=20).json()
    except:
        return None


def scrape_fast():
    """
    Scrape ALL NBA games from the 2025–2026 season start to TODAY
    using ESPN public APIs with date range iteration.
    """

    print("🔍 Fetching NBA Game Logs (ESPN API, full season range)...")

    # 2025–26 expected NBA season start:
    SEASON_START = datetime(2025, 10, 10)   # preseason + early games
    TODAY = datetime.utcnow()

    all_rows = []

    # Iterate day-by-day from season start to today
    date = SEASON_START
    print(f"📅 Fetching games from {SEASON_START.date()} → {TODAY.date()}")

    while date <= TODAY:
        date_str = date.strftime("%Y%m%d")
        print(f"🔎 Checking date: {date_str}")

        events = fetch_scoreboard(date_str)

        if not events:
            date += timedelta(days=1)
            continue

        print(f"  → Found {len(events)} games")

        for game in events:
            game_id = game["id"]
            comps = game.get("competitions", [])
            if not comps:
                continue

            game_date = comps[0]["date"].split("T")[0]

            box = fetch_boxscore(game_id)
            if not box:
                continue

            playersets = box.get("boxscore", {}).get("players", [])

            # team loop
            for teamdata in playersets:
                team_name = teamdata["team"]["displayName"]
                statslist = teamdata.get("statistics", [])

                # player loop
                for p in statslist:
                    player_name = p.get("athlete", {}).get("displayName", "")
                    raw_stats = p.get("stats", [])

                    row = {
                        "GAME_DATE": game_date,
                        "team": team_name,
                        "player_name": player_name,
                        "points": None,
                        "rebounds": None,
                        "assists": None,
                        "minutes": None,
                    }

                    for s in raw_stats:
                        val = s.split(" ")[0]

                        if "PTS" in s:
                            row["points"] = int(val)
                        elif "REB" in s:
                            row["rebounds"] = int(val)
                        elif "AST" in s:
                            row["assists"] = int(val)
                        elif "MIN" in s:
                            try:
                                row["minutes"] = int(val)
                            except:
                                pass

                    all_rows.append(row)

        # next day
        date += timedelta(days=1)

    if not all_rows:
        raise Exception("❌ No NBA game logs returned for the date range.")

    df = pd.DataFrame(all_rows)

    os.makedirs(os.path.join(ROOT_DIR, "data"), exist_ok=True)
    out_path = os.path.join(ROOT_DIR, "data", "player_game_logs_raw.csv")
    df.to_csv(out_path, index=False)

    print(f"Saved RAW logs → {out_path}")
    print(f"Total rows: {len(df)}")

    return df


# AUTO-RUN WHEN CALLED FROM STREAMLIT PIPELINE
if __name__ == "__main__":
    scrape_fast()