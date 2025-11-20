# app/dashboard.py

import streamlit as st
import pandas as pd
import xgboost as xgb
import subprocess
import os
from utils.features import get_feature_columns

st.set_page_config(page_title="NBA Player Predictor", layout="wide")

# -----------------------------
# Utility: Run Python script
# -----------------------------
def run_script(label, script_path):
    st.write(f"### 🔧 {label}")
    output = subprocess.Popen(
        ["python", script_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    stdout, stderr = output.communicate()

    if stdout:
        st.success(stdout.replace("\n", "  \n"))
    if stderr:
        st.error(stderr.replace("\n", "  \n"))


# -----------------------------
# Sidebar Controls
# -----------------------------

st.sidebar.header("⚙️ Pipeline Controls")

if st.sidebar.button("1️⃣ Run Scraper (Download Logs)"):
    run_script("Running Scraper...", "scrapers/scrape_basic_logs.py")

if st.sidebar.button("2️⃣ Build Dataset (Feature Engineering)"):
    run_script("Building Dataset...", "scrapers/build_dataset.py")

if st.sidebar.button("3️⃣ Train Models (PTS/REB/AST)"):
    run_script("Training Models...", "train/train_all.py")


st.sidebar.markdown("---")


# -----------------------------
# Dashboard Data Loading
# -----------------------------

@st.cache_data
def load_dataset():
    if not os.path.exists("data/player_game_logs.csv"):
        st.warning("Dataset not found. Run the pipeline steps first.")
        return pd.DataFrame()
    return pd.read_csv("data/player_game_logs.csv")


@st.cache_resource
def load_model(target):
    path = f"models/{target}_xgb.json"
    if not os.path.exists(path):
        st.warning(f"Model {target} not trained. Run training.")
        return None
    model = xgb.XGBRegressor()
    model.load_model(path)
    return model


df = load_dataset()

if df.empty:
    st.title("🏀 NBA Player Predictor")
    st.info("Run steps 1–3 in the left sidebar to build your system.")
    st.stop()


points_model = load_model("points")
reb_model = load_model("rebounds")
ast_model = load_model("assists")

if None in (points_model, reb_model, ast_model):
    st.title("🏀 NBA Player Predictor")
    st.warning("Models missing. Train them using step 3 in the sidebar.")
    st.stop()


feature_cols = get_feature_columns()


# -----------------------------
# Player Selection
# -----------------------------

st.title("🏀 NBA Player Predictor")

players = sorted(df["player_name"].unique())
player = st.selectbox("Choose a Player", players)

player_data = df[df["player_name"] == player].sort_values("GAME_DATE")
latest = player_data.tail(1).iloc[0]

X = latest[feature_cols].values.reshape(1, -1)

pred_pts = float(points_model.predict(X)[0])
pred_reb = float(reb_model.predict(X)[0])
pred_ast = float(ast_model.predict(X)[0])

col1, col2, col3 = st.columns(3)
col1.metric("Projected Points", f"{pred_pts:.1f}")
col2.metric("Projected Rebounds", f"{pred_reb:.1f}")
col3.metric("Projected Assists", f"{pred_ast:.1f}")

st.markdown("### Last 10 Games")
st.line_chart(
    player_data.tail(10).set_index("GAME_DATE")[["points", "rebounds", "assists"]]
)

st.markdown("### Feature Breakdown")
st.dataframe(latest[feature_cols])
