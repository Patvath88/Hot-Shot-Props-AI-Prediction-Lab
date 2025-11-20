# train/train_model.py

import os
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from utils.features import get_feature_columns


def train_stat_model(target: str, df_path="data/player_game_logs.csv", model_dir="models"):
    """
    Trains an XGBoost regression model for predicting an NBA stat.
    target = "points", "rebounds", "assists", "threes" etc.
    """

    df = pd.read_csv(df_path)

    # drop missing values for target
    df = df.dropna(subset=[target])

    feature_cols = get_feature_columns()

    # ensure all feature columns exist
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise Exception(f"Missing feature columns: {missing}")

    X = df[feature_cols]
    y = df[target]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = xgb.XGBRegressor(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.03,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
        tree_method="hist",
    )

    model.fit(X_train, y_train)

    preds = model.predict(X_val)
    mae = mean_absolute_error(y_val, preds)
    print(f"{target.upper()} MAE: {mae:.3f}")

    os.makedirs(model_dir, exist_ok=True)
    model.save_model(f"{model_dir}/{target}_xgb.json")

    print(f"Saved → {model_dir}/{target}_xgb.json")
    return mae
