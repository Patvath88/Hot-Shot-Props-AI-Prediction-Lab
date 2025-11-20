# app/dashboard.py

import sys
import os
import subprocess
import streamlit as st
import pandas as pd
import xgboost as xgb

# =====================================================================
# FIX PATHS FOR STREAMLIT CLOUD
# =====================================================================
APP_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(APP_DIR, ".."))
sys.path.insert(0, ROOT_DIR)

from utils.features import get_feature_columns


# =====================================================================
# RUN PIPELINE SCRIPTS WITH CORRECT PYTHON INTERPRETER
# =====================================================================
def run_pipeline_step(label, relative_script_path):
    """Runs a pipeline step and streams output live in Streamlit."""
    st.write(f"### 🔧 {label}")

    script_path = os.path.join(ROOT_DIR, relative_script_path)

    try:
        process = subprocess.Popen(
            [sys.executable, script_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        out, err = process.communicate()

        if out:
            st.success(out.replace("\n", "  \n"))

        if err:
            st.error(err.replace("\n", "  \n"))

    except Exception as e:
        st.error(f"❌ Error running pipeline step: {e}")


# =====================================================================
# SIDEBAR PIPELINE CONTROL PANEL
# =====================================================================
st.sidebar.header("⚙️ Unified Pipeline")

# ⭐ FULL PIPELINE (Scrape → Build → Train)
if st.sidebar.button("🔄 Run FULL Pipeline"):
    run_pipeline_step("1️⃣ Scraping Logs", "scrapers/scrape_basic_logs_fast.py")
    run_pipeline_step("2️⃣ Building Dataset", "scrapers/build_dataset.py")
    run_pipeline_step("3️⃣ Training Models", "train/train_all.py")
    st.sidebar.success("🎉 Full pipeline completed!")
    st.rerun()

st.sidebar.markdown("---")

# ⭐ INDIVIDUAL STEPS
if st.sidebar.button("1️⃣ Scrape Logs Only"):
    run_pipeline_step("Scraping Logs", "scrapers/scrape_basic_logs_fast.py")

if st.sidebar.button("2️⃣ Build Dataset Only"):
    run_pipeline_step("Building Dataset", "scrapers/build_dataset.py")

if st.sidebar.button("3️⃣ Train Models Only"):
    run_pipeline_step("Training Models", "train/train_all.py")

st.sidebar.markdown("---")


# =====================================================================
# LOAD DATASET
# =====================================================================
@st.cache_data
def load_dataset():
    path = os.path.join(ROOT_DIR, "data/player_game_logs.csv")
    if not os.path.exists(path):
        st.warning("Dataset missing. Run the full pipeline first.")
        return pd.DataFrame()
    return pd.read_csv(path)


@st.cache_resource
def load_model(target):
    path = os.path.join(ROOT_DIR, f"models/{target}_xgb.json")
    if not os.path.exists(path):
        return None
    model = xgb.XGBRegressor()
    model.load_model(path)
    return model


df = load_dataset()

if df.empty:
    st.title("🏀 NBA Player Predictor")
    st.info("Run the FULL pipeline (left sidebar) to initialize everything.")
    st.stop()

# Load models
points_model = load_model("points")
reb_model = load_model("rebounds")
ast_model = load_model("assists")

if None in (points_model, reb_model, ast_model):
    st.title("🏀 NBA Player Predictor")
    st.warning("Models missing — run TRAIN MODELS (Step 3) in the sidebar.")
    st.stop()

feature_cols = get_feature_columns()


# =====================================================================
# MAIN UI — PLAYER PROJECTIONS
# =====================================================================
st.title("🏀 NBA Player Stat Predictor")

players = sorted(df["player_name"].unique())
player = st.selectbox("Select Player", players)

player_df = df[df["player_name"] == player].sort_values("GAME_DATE")
latest = player_df.tail(1).iloc[0]

X = latest[feature_cols].values.reshape(1, -1)

pred_pts = float(points_model.predict(X)[0])
pred_reb = float(reb_model.predict(X)[0])
pred_ast = float(ast_model.predict(X)[0])

col1, col2, col3 = st.columns(3)
col1.metric("Projected Points", f"{pred_pts:.1f}")
col2.metric("Projected Rebounds", f"{pred_reb:.1f}")
col3.metric("Projected Assists", f"{pred_ast:.1f}")

st.markdown("---")

# Trend chart
st.subheader("📈 Last 10 Games")
st.line_chart(
    player_df.tail(10).set_index("GAME_DATE")[["points", "rebounds", "assists"]]
)

# Detailed features
st.subheader("🔍 Features Used for Prediction")
st.dataframe(latest[feature_cols], use_container_width=True)
