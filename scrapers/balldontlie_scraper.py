# scrapers/balldontlie_scraper.py

import requests
import pandas as pd
from tqdm import tqdm
import time

BASE = "https://api.balldontlie.io/v1"

def get_players():
    players = []
    page = 1

    while True:
        r = requests.get(f"{BASE}/players", params={"page": page, "per_page": 100})
        data = r.json()
        players += data["data"]
        if data["meta"]["next_page"] is None:
            break
        page += 1
        time.sleep(0.2)

    df = pd.DataFrame(players)
    df.to_csv("data/bdl_players.csv", index=False)
    return df


def get_player_game_logs(season=2024):
    players = pd.read_csv("data/bdl_players.csv")

    all_logs = []

    for _, p in tqdm(players.iterrows(), total=len(players)):
        pid = p["id"]

        page = 1
        while True:
            r = requests.get(f"{BASE}/stats", params={
                "player_ids[]": pid,
                "seasons[]": season,
                "page": page,
                "per_page": 100
            })
            d = r.json()
            all_logs += d["data"]

            if d["meta"]["next_page"] is None:
                break
            page += 1
            time.sleep(0.2)

    df = pd.DataFrame(all_logs)
    df.to_csv("data/bdl_game_logs.csv", index=False)
    return df


if __name__ == "__main__":
    get_players()
    get_player_game_logs()
