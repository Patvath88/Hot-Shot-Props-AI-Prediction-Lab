# train/train_models.py
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import xgboost as xgb

from utils.features import add_basic_features, get_feature_columns
from utils.io_utils import load_game_logs, save_model


RANDOM_STATE = 42
MODELS_DIR = "models"


def train_target(df, target_col: str, model_name: str):
    df = df.dropna(subset=[target_col])
    feature_cols = get_feature_columns()

    X = df[feature_cols]
    y = df[target_col]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    # XGBoost – balanced for accuracy & speed
    model = xgb.XGBRegressor(
        n_estimators=400,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        random_state=RANDOM_STATE,
        tree_method="hist",   # fast
    )

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="mae",
        verbose=False,
    )

    val_pred = model.predict(X_val)
    mae = mean_absolute_error(y_val, val_pred)
    print(f"{model_name} MAE: {mae:.3f}")

    save_path = os.path.join(MODELS_DIR, f"{model_name}.json")
    save_model(model, save_path)
    print(f"Saved {model_name} to {save_path}")


def main():
    df = load_game_logs()
    df = add_basic_features(df)

    # Ensure no weird inf/nan
    df = df.replace([float("inf"), float("-inf")], pd.NA).dropna()

    # Train separate models
    train_target(df, "points", "points_xgb")
    train_target(df, "rebounds", "rebounds_xgb")
    train_target(df, "assists", "assists_xgb")


if __name__ == "__main__":
    main()
