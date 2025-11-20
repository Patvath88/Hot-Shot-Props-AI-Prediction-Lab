# scrapers/nba_api_scraper.py

import pandas as pd
from nba_api.stats.endpoints import teamdashboardbygeneralsplits, leaguedashplayerstats
from nba_api.stats.endpoints import playervsplayer
import time


def get_player_advanced(season="2024-25"):
    df = leaguedashplayerstats.LeagueDashPlayerStats(
        season=season,
        measure_type_detailed_defense="Advanced"
    ).get_data_frames()[0]

    df.to_csv("data/nba_player_advanced.csv", index=False)
    return df


def get_team_advanced(season="2024-25"):
    df = teamdashboardbygeneralsplits.TeamDashboardByGeneralSplits(
        season=season
    ).get_data_frames()[0]

    df.to_csv("data/nba_team_advanced.csv", index=False)
    return df


if __name__ == "__main__":
    get_player_advanced()
    get_team_advanced()
