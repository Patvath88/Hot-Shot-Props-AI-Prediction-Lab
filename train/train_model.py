# train/train_model.py

import sys, os
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
sys.path.insert(0, ROOT_DIR)

import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from utils.features import get_feature_columns


def train_stat_model(target):
    print(f"🎯 Training {target.upper()} model...")

    df = pd.read_csv(os.path.join(ROOT_DIR, "data/player_game_logs.csv"))
    df = df.dropna(subset=[target])

    X = df[get_feature_columns()]
    y = df[target]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = xgb.XGBRegressor(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.04,
        subsample=0.9,
        colsample_bytree=0.9,
        tree_method="hist",
        random_state=42
    )

    model.fit(X_train, y_train)
    preds = model.predict(X_val)
    mae = mean_absolute_error(y_val, preds)

    print(f"📉 {target.upper()} MAE: {mae:.3f}")

    # FIXED MODEL SAVE PATH
    model_dir = os.path.join(ROOT_DIR, "models")
    os.makedirs(model_dir, exist_ok=True)

    model_path = os.path.join(model_dir, f"{target}_xgb.json")
    model.save_model(model_path)

    print(f"💾 Saved → {model_path}")