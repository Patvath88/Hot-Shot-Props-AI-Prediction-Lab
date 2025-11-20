# utils/features.py

import pandas as pd
import numpy as np


ROLL = 5


def add_basic_features(df):
    df = df.sort_values(["player_name", "GAME_DATE"])

    g = df.groupby("player_name")

    # Rolling averages
    df["points_rolling_5"] = g["points"].rolling(ROLL, min_periods=1).mean().reset_index(0, drop=True)
    df["rebounds_rolling_5"] = g["rebounds"].rolling(ROLL, min_periods=1).mean().reset_index(0, drop=True)
    df["assists_rolling_5"] = g["assists"].rolling(ROLL, min_periods=1).mean().reset_index(0, drop=True)
    df["minutes_rolling"] = g["minutes"].rolling(ROLL, min_periods=1).mean().reset_index(0, drop=True)

    # Season averages
    df["points_avg_season"] = g["points"].transform("mean")
    df["rebounds_avg_season"] = g["rebounds"].transform("mean")
    df["assists_avg_season"] = g["assists"].transform("mean")
    df["minutes_avg_season"] = g["minutes"].transform("mean")

    # Form score (simple weighted last-5)
    df["form_score"] = (
        df["points_rolling_5"] * 0.5 +
        df["rebounds_rolling_5"] * 0.3 +
        df["assists_rolling_5"] * 0.2
    )

    return df


def get_feature_columns():
    """
    Features used by XGBoost models.
    """
    return [
        # rolling
        "points_rolling_5",
        "rebounds_rolling_5",
        "assists_rolling_5",
        "minutes_rolling",

        # season
        "points_avg_season",
        "rebounds_avg_season",
        "assists_avg_season",
        "minutes_avg_season",

        # minutes & form
        "minutes",
        "form_score",
    ]
