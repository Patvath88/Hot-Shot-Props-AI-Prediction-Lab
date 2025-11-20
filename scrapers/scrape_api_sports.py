import requests
import pandas as pd
import os
from config import API_SPORTS_KEY, API_SPORTS_URL

HEADERS = {
    "x-apisports-key": API_SPORTS_KEY
}

def fetch_game_logs(season="2025-2026"):
    print(f"📡 Fetching API-Sports NBA logs for {season}...")

    all_rows = []
    page = 1
    more = True

    while more:
        url = f"{API_SPORTS_URL}/games"
        params = {
            "league": "12",          # NBA
            "season": season,
            "page": page
        }

        resp = requests.get(url, headers=HEADERS, params=params).json()

        if "response" not in resp or len(resp["response"]) == 0:
            more = False
            break

        for game in resp["response"]:
            game_id = game["id"]
            date = game["date"].split("T")[0]

            # Fetch boxscore for this game
            box_url = f"{API_SPORTS_URL}/games/statistics"
            box = requests.get(box_url, headers=HEADERS, params={"game": game_id}).json()

            if "response" not in box:
                continue

            for team in box["response"]:
                team_name = team["team"]["name"]

                for p in team["players"]:
                    if "points" not in p["statistics"]:
                        continue

                    stats = p["statistics"]

                    row = {
                        "GAME_DATE": date,
                        "team": team_name,
                        "player_name": p["player"]["name"],
                        "points": stats["points"],
                        "rebounds": stats["totReb"],
                        "assists": stats["assists"],
                        "minutes": stats["min"],
                        "fg_pct": stats["fgp"],
                        "fg3_pct": stats["tpp"],
                        "ft_pct": stats["ftp"],
                        "steals": stats["steals"],
                        "blocks": stats["blocks"],
                        "turnovers": stats["turnovers"]
                    }
                    all_rows.append(row)

        page += 1

    df = pd.DataFrame(all_rows)

    os.makedirs("data", exist_ok=True)
    df.to_csv("data/player_game_logs_raw.csv", index=False)

    print(f"✅ Saved RAW logs → data/player_game_logs_raw.csv")
    print(f"📈 Total rows: {len(df)}")
    return df


if __name__ == "__main__":
    fetch_game_logs()
