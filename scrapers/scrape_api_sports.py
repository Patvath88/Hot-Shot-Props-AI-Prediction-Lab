import requests
import pandas as pd
import os

API_KEY = "cd765274223af1fd1878cdeb02d8aa9b"
BASE = "https://v1.basketball.api-sports.io/"

HEADERS = {
    "cd765274223af1fd1878cdeb02d8aa9b": API_KEY
}

SEASON = "2025-2026"


def get(endpoint, params):
    r = requests.get(BASE + endpoint, headers=HEADERS, params=params)
    r.raise_for_status()
    return r.json()["response"]


def scrape_api_sports():
    print("🔍 Fetching API-Sports NBA Stats...")

    games = get("games", {"season": SEASON, "league": 12})
    all_rows = []

    for g in games:
        game_id = g["id"]
        date = g["date"].split("T")[0]

        stats = get("games/statistics", {"id": game_id})

        for team in stats:
            players = team["players"]
            team_name = team["team"]["name"]

            for p in players:
                row = {
                    "GAME_ID": game_id,
                    "GAME_DATE": date,
                    "team": team_name,
                    "player_id": p["player"]["id"],
                    "player_name": p["player"]["name"],
                    "points": p["statistics"]["points"],
                    "rebounds": p["statistics"]["rebounds"],
                    "assists": p["statistics"]["assists"],
                    "minutes": p["statistics"]["minutes"]
                }

                all_rows.append(row)

    df = pd.DataFrame(all_rows)
    os.makedirs("data", exist_ok=True)
    df.to_csv("data/player_game_logs_raw.csv", index=False)

    print("Saved → data/player_game_logs_raw.csv")
    print("Rows:", len(df))
    return df


if __name__ == "__main__":
    scrape_api_sports()
