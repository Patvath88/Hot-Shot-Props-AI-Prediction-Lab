import streamlit as st
import pandas as pd
import xgboost as xgb
import os
import subprocess
def get_feature_columns():
    return [
        "points_last5",
        "rebounds_last5",
        "assists_last5",
        "minutes_last5",
        "points_avg",
        "rebounds_avg",
        "assists_avg",
        "minutes_avg",
        "usage_rate",
        "form_score"
    ]

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

st.title("🏀 NBA Predictor (API-Sports Edition)")

def run_step(name, script_path):
    st.write(f"### {name}")
    out = subprocess.run(
        ["python", os.path.join(ROOT, script_path)],
        capture_output=True, text=True
    )
    st.text(out.stdout)
    st.error(out.stderr)

if st.sidebar.button("Run FULL Pipeline"):
    run_step("Scraping API-Sports", "scrapers/scrape_api_sports.py")
    run_step("Building Dataset", "scrapers/build_dataset.py")
    run_step("Training Models", "train/train_all.py")

# Load dataset
if os.path.exists("data/player_game_logs.csv"):
    df = pd.read_csv("data/player_game_logs.csv")
else:
    st.warning("Run pipeline first")
    st.stop()

# Load models
def load_model(name):
    path = f"models/{name}_xgb.json"
    if not os.path.exists(path):
        return None
    m = xgb.XGBRegressor()
    m.load_model(path)
    return m

points_model = load_model("points")
reb_model = load_model("rebounds")
ast_model = load_model("assists")

if points_model is None:
    st.warning("Train the models first.")
    st.stop()

player = st.selectbox("Player", sorted(df["player_name"].unique()))
latest = df[df["player_name"] == player].tail(1)

X = latest[get_feature_columns()].values

st.metric("Predicted Points", f"{points_model.predict(X)[0]:.1f}")
st.metric("Predicted Rebounds", f"{reb_model.predict(X)[0]:.1f}")
st.metric("Predicted Assists", f"{ast_model.predict(X)[0]:.1f}")
