# scrapers/espn_scraper.py

import requests
import pandas as pd

def get_espn_injuries():
    url = "https://site.web.api.espn.com/apis/v2/sports/basketball/nba/injuries"
    data = requests.get(url).json()

    rows = []
    for team in data["injuries"]:
        for entry in team["injuries"]:
            rows.append({
                "player_name": entry["athlete"]["displayName"],
                "status": entry["status"],
                "type": entry["type"],
                "date": entry["date"]
            })

    df = pd.DataFrame(rows)
    df.to_csv("data/espn_injuries.csv", index=False)
    return df
