# train/train_model.py

import os
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from utils.features import get_feature_columns


def train_stat_model(target, dataset_path="data/player_game_logs.csv"):
    df = pd.read_csv(dataset_path)

    df = df.dropna(subset=[target])

    feature_cols = get_feature_columns()

    X = df[feature_cols]
    y = df[target]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = xgb.XGBRegressor(
        n_estimators=350,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
        tree_method="hist"
    )

    model.fit(X_train, y_train)
    preds = model.predict(X_val)
    mae = mean_absolute_error(y_val, preds)

    print(f"{target.upper()} MAE: {mae:.3f}")

    os.makedirs("models", exist_ok=True)
    model.save_model(f"models/{target}_xgb.json")

    print(f"Saved → models/{target}_xgb.json")
    return mae
