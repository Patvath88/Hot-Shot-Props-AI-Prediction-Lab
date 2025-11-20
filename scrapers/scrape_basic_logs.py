# scrapers/scrape_basic_logs.py

import sys, os
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
sys.path.insert(0, ROOT_DIR)

import pandas as pd
import time
from tqdm import tqdm
from nba_api.stats.static import players
from nba_api.stats.endpoints import playergamelog


def scrape_basic_game_logs(season="2024-25"):

    all_players = players.get_active_players()
    rows = []

    for p in tqdm(all_players, desc="Scraping player logs"):
        pid = p["id"]
        name = p["full_name"]

        try:
            logs = playergamelog.PlayerGameLog(
                player_id=pid, season=season
            ).get_data_frames()[0]
            logs["player_name"] = name
            rows.append(logs)
        except Exception as e:
            print("ERROR:", name, e)

        time.sleep(0.4)

    df = pd.concat(rows, ignore_index=True)

    df.rename(columns={
        "PTS": "points",
        "REB": "rebounds",
        "AST": "assists",
        "MIN": "minutes"
    }, inplace=True)

    df.to_csv("data/player_game_logs_raw.csv", index=False)
    print("Saved raw logs → data/player_game_logs_raw.csv")
    print("Rows:", len(df))


if __name__ == "__main__":
    scrape_basic_game_logs()