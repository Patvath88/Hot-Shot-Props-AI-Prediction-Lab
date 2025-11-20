import pandas as pd
import xgboost as xgb
import os
from utils.features import get_feature_columns

def train_stat_model(target):
    df = pd.read_csv("data/player_game_logs.csv")

    X = df[get_feature_columns()]
    y = df[target]

    model = xgb.XGBRegressor(
        n_estimators=300,
        learning_rate=0.07,
        max_depth=6
    )
    model.fit(X, y)

    os.makedirs("models", exist_ok=True)
    model.save_model(f"models/{target}_xgb.json")

    print(f"Saved: models/{target}_xgb.json")