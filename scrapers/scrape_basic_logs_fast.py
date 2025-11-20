# scrapers/scrape_basic_logs_fast.py

import sys, os, requests, pandas as pd

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
sys.path.insert(0, ROOT_DIR)


def scrape_fast(season="2024-25"):
    """
    Fast, Streamlit-safe scraper using BallDontLie API.
    """

    print("🔍 Fetching game logs from BallDontLie API...")
    base_url = "https://api.balldontlie.io/v1/stats"

    all_rows = []
    page = 1

    headers = {
        "Authorization": "",  # leave blank for free tier
    }

    season_year = season.split("-")[0]

    while True:
        url = f"{base_url}?seasons[]={season_year}&per_page=100&page={page}"
        print(f"Fetching page {page}...")
        resp = requests.get(url, headers=headers)

        if resp.status_code != 200:
            print("❌ Error fetching page", page)
            break

        data = resp.json()
        rows = data.get("data", [])

        if not rows:
            print("No more pages.")
            break

        all_rows.extend(rows)
        page += 1

    print(f"Total logs fetched: {len(all_rows)}")

    # Convert to DataFrame
    df = pd.json_normalize(all_rows)

    # --- FIX: Correct column names ---
    df = df.rename(columns={
        "pts": "points",
        "reb": "rebounds",
        "ast": "assists",
        "min": "minutes",
        "game.date": "GAME_DATE",
    })

    # --- FIX: Use correct JSON fields ---
    df["player_name"] = (
        df["player.first_name"].fillna("") + " " + df["player.last_name"].fillna("")
    ).str.strip()

    df.to_csv("data/player_game_logs_raw.csv", index=False)
    print("Saved RAW logs → data/player_game_logs_raw.csv")

    return df


if __name__ == "__main__":
    scrape_fast()
