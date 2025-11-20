# utils/features.py
import pandas as pd
import numpy as np

ROLLING_WINDOW = 5

def add_basic_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    df: player game logs with at least:
        ['player_id', 'game_date', 'team', 'opp_team',
         'minutes', 'points', 'rebounds', 'assists',
         'fg_attempts', 'threes_attempts', 'usage_rate', 'pace']
    """
    df = df.sort_values(["player_id", "game_date"]).copy()

    group = df.groupby("player_id")

    for stat in ["points", "rebounds", "assists"]:
        df[f"{stat}_avg_season"] = group[stat].transform("mean")
        df[f"{stat}_rolling_{ROLLING_WINDOW}"] = (
            group[stat].transform(lambda x: x.rolling(ROLLING_WINDOW, min_periods=1).mean())
        )
        df[f"{stat}_std_rolling_{ROLLING_WINDOW}"] = (
            group[stat].transform(lambda x: x.rolling(ROLLING_WINDOW, min_periods=1).std())
        )

    # Minutes
    df["minutes_avg_season"] = group["minutes"].transform("mean")
    df["minutes_rolling"] = group["minutes"].transform(
        lambda x: x.rolling(ROLLING_WINDOW, min_periods=1).mean()
    )

    # Simple recent form metric
    df["form_score"] = (
        0.5 * df["points_rolling_5"].fillna(0)
        + 0.3 * df["rebounds_rolling_5"].fillna(0)
        + 0.2 * df["assists_rolling_5"].fillna(0)
    )

    # Opponent defensive indicators (assume pre-merged)
    # e.g. opp_def_rating, opp_pace, opp_reb_allowed, etc.

    return df


def get_feature_columns():
    # central place for features used by all models
    return [
        "minutes_avg_season",
        "minutes_rolling",
        "points_avg_season",
        "points_rolling_5",
        "points_std_rolling_5",
        "rebounds_avg_season",
        "rebounds_rolling_5",
        "rebounds_std_rolling_5",
        "assists_avg_season",
        "assists_rolling_5",
        "assists_std_rolling_5",
        "usage_rate",
        "pace",
        "form_score",
        "opp_def_rating",
        "opp_pace",
        "opp_points_allowed",
        "opp_reb_allowed",
        "opp_ast_allowed",
    ]
