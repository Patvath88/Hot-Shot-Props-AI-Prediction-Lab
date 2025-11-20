# utils/io_utils.py
import os
import json
import pandas as pd
import xgboost as xgb

def load_game_logs(path: str = "data/player_game_logs.csv") -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=["game_date"])

def save_model(model: xgb.XGBRegressor, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    model.save_model(path)

def load_model(path: str) -> xgb.XGBRegressor:
    model = xgb.XGBRegressor()
    model.load_model(path)
    return model
