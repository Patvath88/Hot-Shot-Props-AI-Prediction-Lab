from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import requests
import pandas as pd
from datetime import datetime, timedelta

app = FastAPI()

# Allow all origins (public API)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

NBA_BASE = "https://stats.nba.com/stats"
HEADERS = {
    "Host": "stats.nba.com",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
    "Accept": "application/json, text/plain, */*",
    "Referer": "https://www.nba.com/",
    "Origin": "https://www.nba.com",
    "Connection": "keep-alive"
}


def get_nba_json(endpoint, params):
    r = requests.get(f"{NBA_BASE}/{endpoint}", headers=HEADERS, params=params)
    r.raise_for_status()
    return r.json()


# ----------------------------------------------------------
# 1) FULL GAME LOGS (season)
# ----------------------------------------------------------
@app.get("/gamelogs")
def gamelogs(season: str = "2025-26"):
    resp = get_nba_json("leaguegamelog", {
        "Season": season,
        "SeasonType": "Regular Season",
        "PlayerOrTeam": "P",
        "Counter": 5000,
        "Sorter": "DATE",
        "Direction": "ASC",
        "LeagueID": "00"
    })
    return resp


# ----------------------------------------------------------
# 2) PLAYER GAME LOG
# ----------------------------------------------------------
@app.get("/playergamelog")
def player_log(player_id: str, season: str = "2025-26"):
    resp = get_nba_json("playergamelog", {
        "PlayerID": player_id,
        "Season": season,
        "SeasonType": "Regular Season",
        "Counter": 500,
        "Direction": "ASC"
    })
    return resp


# ----------------------------------------------------------
# 3) ADVANCED TEAM STATS (ORtg, DRtg, pace)
# ----------------------------------------------------------
@app.get("/team_ratings")
def team_ratings(season: str = "2025-26"):
    resp = get_nba_json("leaguedashteamstats", {
        "Season": season,
        "SeasonType": "Regular Season",
        "MeasureType": "Advanced",
        "PerMode": "PerGame"
    })
    return resp


# ----------------------------------------------------------
# 4) INJURY REPORTS (NBA + ESPN)
# ----------------------------------------------------------
@app.get("/injuries")
def injuries():
    espn = requests.get(
        "https://site.api.espn.com/apis/v2/sports/basketball/nba/injuries"
    ).json()

    out = []
    for team in espn.get("injuries", []):
        for player in team.get("players", []):
            out.append({
                "team": team["team"]["displayName"],
                "player": player["athlete"]["displayName"],
                "status": player["status"],
                "description": player.get("description", ""),
                "last_update": player.get("lastUpdate", "")
            })
    return out


# ----------------------------------------------------------
# 5) RECENT GAMES (last 3 days)
# ----------------------------------------------------------
@app.get("/recent")
def recent():
    today = datetime.utcnow().date()
    out = []

    for i in range(3):
        d = today - timedelta(days=i)
        ds = d.strftime("%Y%m%d")

        url = f"https://site.api.espn.com/apis/v2/sports/basketball/nba/scoreboard"
        data = requests.get(url, params={"dates": ds}).json()

        out.extend(data.get("events", []))

    return out


# ----------------------------------------------------------
# 6) TEAM SCHEDULE
# ----------------------------------------------------------
@app.get("/schedule")
def schedule(season: str = "2025-26"):
    url = f"https://data.nba.com/data/10s/v2015/json/mobile_teams/nba/{season.replace('-', '')}/league/00_full_schedule.json"
    try:
        return requests.get(url).json()
    except:
        return {"error": "Schedule not available"}
