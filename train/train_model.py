import pandas as pd
import xgboost as xgb
import os

FEATURES = [
    "points_rolling_5", "rebounds_rolling_5", "assists_rolling_5",
    "minutes_rolling_5", "form_score"
]

def train_stat_model(target):
    df = pd.read_csv("data/player_game_logs.csv")

    X = df[FEATURES]
    y = df[target]

    dtrain = xgb.DMatrix(X, label=y)

    params = {
        "objective": "reg:squarederror",
        "eta": 0.05,
        "max_depth": 4,
    }

    model = xgb.train(params, dtrain, num_boost_round=150)

    os.makedirs("models", exist_ok=True)
    model.save_model(f"models/{target}_xgb.json")

    print(f"✅ Trained model: {target}")


if __name__ == "__main__":
    train_stat_model("points")