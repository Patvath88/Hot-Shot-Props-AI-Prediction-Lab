# app/dashboard.py

import sys
import os
import subprocess
import streamlit as st
import pandas as pd
import xgboost as xgb

# =============================================
# PATH FIX FOR STREAMLIT CLOUD
# =============================================
APP_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(APP_DIR, ".."))
sys.path.insert(0, ROOT_DIR)

from utils.features import get_feature_columns


# =============================================
# PIPELINE RUNNER
# =============================================
def run_pipeline_step(label, relative_script_path):
    """Run a script inside Streamlit with correct Python interpreter."""
    st.write(f"### 🔧 {label}")

    script_path = os.path.join(ROOT_DIR, relative_script_path)

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
        st.error(f"❌ Pipeline error: {e}")


# =============================================
# SIDEBAR PIPELINE CONTROL PANEL
# =============================================
st.sidebar.header("⚙️ Unified Pipeline")

if st.sidebar.button("🔄 Run FULL Pipeline (Scrape → Build → Train)"):
    run_pipeline_step("1️⃣ Scraping Logs", "scrapers/scrape_basic_logs.py")
    run_pipeline_step("2️⃣ Building Dataset", "scrapers/build_dataset.py")
    run_pipeline_step("3️⃣ Training Models", "train/train_all.py")
    st.sidebar.success("🎉 Full Pipeline Completed!")
    st.experimental_rerun()

st.sidebar.markdown("---")

if st.sidebar.button("1️⃣ Scrape Logs Only"):
    run_pipeline_step("Scraping Logs", "scrapers/scrape_basic_logs.py")

if st.sidebar.button("2️⃣ Build Dataset Only"):
    run_pipeline_step("Building Dataset", "scrapers/build_dataset.py")

if st.sidebar.button("3️⃣ Train Models Only"):
    run_pipeline_step("Training Models", "train/train_all.py")

st.sidebar.markdown("---")


# =============================================
# DATA LOADING
# =============================================
@st.cache_data
def load_dataset():
    path = os.path.join(ROOT_DIR, "data/player_game_logs.csv")
    if not os.path.exists(path):
        st.warning("Dataset missing. Run pipeline first.")
        return pd.DataFrame()
    return pd.read_csv(path)


@st.cache_resource
def load_model(target):
    path = os.path.join(ROOT_DIR, f"models/{target}_xgb.json")
    if not os.path.exists(path):
        return None
    model = xgboost.XGBRegressor()
    model.load_model(path)
    return model


df = load_dataset()

if df.empty:
    st.title("🏀 NBA Player Predictor")
    st.info("Run the Full Pipeline in the sidebar to initialize the system.")
    st.stop()

points_model = load_model("points")
reb_model = load_model("rebounds")
ast_model = load_model("assists")

if None in (points_model, reb_model, ast_model):
    st.warning("Models missing. Train them using Step 3.")
    st.stop()

feature_cols = get_feature_columns()


# =============================================
# MAIN UI — PLAYER PREDICTIONS
# =============================================
st.title("🏀 NBA Player Predictor")

players = sorted(df["player_name"].unique())
player = st.selectbox("Select Player", players)

player_df = df[df["player_name"] == player].sort_values("GAME_DATE")
latest = player_df.tail(1).iloc[0]

X = latest[feature_cols].values.reshape(1, -1)

pred_pts = points_model.predict(X)[0]
pred_reb = reb_model.predict(X)[0]
pred_ast = ast_model.predict(X)[0]

col1, col2, col3 = st.columns(3)
col1.metric("Projected Points", f"{pred_pts:.1f}")
col2.metric("Projected Rebounds", f"{pred_reb:.1f}")
col3.metric("Projected Assists", f"{pred_ast:.1f}")

st.markdown("---")

st.subheader("📈 Last 10 Games")
st.line_chart(
    player_df.tail(10).set_index("GAME_DATE")[["points", "rebounds", "assists"]]
)

st.subheader("🔍 Features Used")
st.dataframe(latest[feature_cols])
