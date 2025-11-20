# app/dashboard.py

import sys
import os
import subprocess
import streamlit as st
import pandas as pd
import xgboost as xgb

# ============================================================
#  PATH FIX FOR STREAMLIT CLOUD
# ============================================================
APP_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(APP_DIR, ".."))
sys.path.insert(0, ROOT_DIR)

from utils.features import get_feature_columns


# ============================================================
#  RUN PYTHON SCRIPTS INSIDE STREAMLIT (WITH CORRECT PYTHON)
# ============================================================
def run_script(label, script_path):
    st.write(f"### 🔧 {label}")

    try:
        result = subprocess.Popen(
            [sys.executable, script_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        out, err = result.communicate()

        if out:
            st.success(out.replace("\n", "  \n"))

        if err:
            st.error(err.replace("\n", "  \n"))

    except Exception as e:
        st.error(f"❌ Failed to run script: {e}")


# ============================================================
#  SIDEBAR PIPELINE CONTROLS
# ============================================================
st.sidebar.header("⚙️ Pipeline Controls")

if st.sidebar.button("1️⃣ Run Scraper (Download Logs)"):
    run_script("Running scraper...", "scrapers/scrape_basic_logs.py")

if st.sidebar.button("2️⃣ Build Dataset (Feature Engineering)"):
    run_script("Building dataset...", "scrapers/build_dataset.py")

if st.sidebar.button("3️⃣ Train Models (PTS/REB/AST)"):
    run_script("Training models...", "train/train_all.py")

st.sidebar.markdown("---")


# ============================================================
#  LOAD DATA & MODELS
# ============================================================
@st.cache_data
def load_dataset():
    path = "data/player_game_logs.csv"
    if not os.path.exists(path):
        st.warning("Dataset not found. Run scraper + dataset builder first.")
        return pd.DataFrame()
    return pd.read_csv(path)


@st.cache_resource
def load_model(target):
    path = f"models/{target}_xgb.json"
    if not os.path.exists(path):
        return None

    model = xgboost.XGBRegressor()
    model.load_model(path)
    return model


df = load_dataset()

if df.empty:
    st.title("🏀 NBA Player Predictor")
    st.info("Run the pipeline steps in the sidebar to initialize the system.")
    st.stop()

points_model = load_model("points")
reb_model = load_model("rebounds")
ast_model = load_model("assists")

if None in (points_model, reb_model, ast_model):
    st.warning("Models missing. Train them using Step 3.")
    st.stop()

feature_cols = get_feature_columns()


# ============================================================
#  MAIN PLAYER PREDICTION UI
# ============================================================
st.title("🏀 NBA Player Predictor")

players = sorted(df["player_name"].unique())
player = st.selectbox("Choose a Player", players)

player_df = df[df["player_name"] == player].sort_values("GAME_DATE")
latest = player_df.tail(1).iloc[0]

X = latest[feature_cols].values.reshape(1, -1)

pred_points = float(points_model.predict(X)[0])
pred_rebounds = float(reb_model.predict(X)[0])
pred_assists = float(ast_model.predict(X)[0])

col1, col2, col3 = st.columns(3)
col1.metric("Projected Points", f"{pred_points:.1f}")
col2.metric("Projected Rebounds", f"{pred_rebounds:.1f}")
col3.metric("Projected Assists", f"{pred_assists:.1f}")

st.markdown("---")

st.subheader("📈 Last 10 Games")
st.line_chart(
    player_df.tail(10).set_index("GAME_DATE")[["points", "rebounds", "assists"]]
)

st.markdown("### Feature Breakdown")
st.dataframe(latest[feature_cols], use_container_width=True)
