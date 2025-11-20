# build_player_game_dataset.py

import pandas as pd
from utils.features import add_basic_features


def build_dataset():
    bdl = pd.read_csv("data/bdl_game_logs.csv")
    espn = pd.read_csv("data/espn_injuries.csv")
    adv = pd.read_csv("data/nba_player_advanced.csv")

    # simple merges
    df = bdl.merge(adv, left_on="player.id", right_on="PLAYER_ID", how="left")
    df = df.merge(espn, left_on="player_name", right_on="player_name", how="left")

    # rename key stats
    df = df.rename(columns={
        "pts": "points",
        "reb": "rebounds",
        "ast": "assists",
        "min": "minutes",
    })

    # add rolling + form features
    df = add_basic_features(df)

    df.to_csv("data/player_game_logs.csv", index=False)
    print("Saved → data/player_game_logs.csv")
    return df


if __name__ == "__main__":
    build_dataset()
