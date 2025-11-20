import sys, os, requests, pandas as pd

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
sys.path.insert(0, ROOT_DIR)


def scrape_fast():
    """
    Scrape **CURRENT NBA GAME LOGS** from ESPN's public API.
    Works for 2025–2026 season + pulls most recent games.
    """

    print("🔍 Fetching NBA Game Logs (ESPN API)...")

    all_rows = []

    # ESPN API for scoreboard (gets LAST ~7 days)
    scoreboard_url = "https://site.api.espn.com/apis/v2/sports/basketball/nba/scoreboard"

    scoreboard = requests.get(scoreboard_url, timeout=20).json()
    events = scoreboard.get("events", [])

    print(f"Found {len(events)} games...")

    for game in events:
        game_id = game["id"]

        box_url = f"https://site.api.espn.com/apis/v2/sports/basketball/nba/summary?event={game_id}"
        box = requests.get(box_url, timeout=20).json()

        competitions = game.get("competitions", [])
        if not competitions:
            continue

        date = competitions[0]["date"].split("T")[0]

        # Get team-level player stats
        for team in box.get("boxscore", {}).get("players", []):
            team_name = team["team"]["displayName"]

            for p in team.get("statistics", []):
                player_name = p.get("athlete", {}).get("displayName", "")
                stats = p.get("stats", [])

                row = {
                    "GAME_DATE": date,
                    "team": team_name,
                    "player_name": player_name,
                }

                # stats are dynamic strings like "23 PTS", "10 REB"
                for s in stats:
                    if "PTS" in s:
                        row["points"] = int(s.split(" ")[0])
                    if "REB" in s:
                        row["rebounds"] = int(s.split(" ")[0])
                    if "AST" in s:
                        row["assists"] = int(s.split(" ")[0])
                    if "MIN" in s:
                        mins = s.split(" ")[0]
                        try:
                            row["minutes"] = int(mins)
                        except:
                            row["minutes"] = None

                all_rows.append(row)

    if not all_rows:
        raise Exception("❌ No game logs found. ESPN API may be empty today.")

    df = pd.DataFrame(all_rows)

    os.makedirs("data", exist_ok=True)
    df.to_csv("data/player_game_logs_raw.csv", index=False)

    print("Saved RAW logs → data/player_game_logs_raw.csv")
    print("Rows:", len(df))

    return df


# ⭐ MAKE SCRAPER RUN WHEN FILE IS EXECUTED
if __name__ == "__main__":
    scrape_fast()