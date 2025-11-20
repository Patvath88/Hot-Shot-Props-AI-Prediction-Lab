# app/dashboard.py

# ============================================================
#  PATH FIX FOR STREAMLIT & STREAMLIT CLOUD
# ============================================================
import sys
import os

# This ensures Streamlit can import utils/, scrapers/, train/
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT_DIR)
sys.path.append(os.path.abspath(os.path.join(ROOT_DIR, "..")))

# ============================================================
#  IMPORTS
# ============================================================
import streamlit as st
import pandas as pd
import xgboost as xgb
import subprocess
from utils.features import get_feature_columns


# ============================================================
#  STREAMLIT SETTINGS
# ============================================================
st.set_page_config(page_title="NBA Player Predictor", layout="wide")

# ============================================================
#  HELPERS — RUN PYTHON SCRIPTS INSIDE STREAMLIT
# ============================================================
def run_script(label, script_path):
    """Runs a local python file & prints output inside Streamlit."""
    st.write(f"### 🔧 {label}")
    try:
        result = subprocess.Popen(
            ["python", script_path],
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
        st.error(f"Failed to run script: {e}")


# ============================================================
#  SIDEBAR — PIPELINE CONTROLS
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
#  LOAD DATASET + MODELS
# ============================================================
@st.cache_data
def load_dataset():
    path = "data/player_game_logs.csv"
    if not os.path.exists(path):
        st.warning("Dataset missing. You must run steps 1–2 first.")
        return pd.DataFrame()
    return pd.read_csv(path)


@st.cache_resource
def load_model(target):
    """Loads an XGBoost model safely."""
    path = f"models/{target}_xgb.json"
    if not os.path.exists(path):
        return None
    model = xgb.XGBRegressor()
    model.load_model(path)
    return model


df = load_dataset()

# If dataset is empty, stop here
if df.empty:
    st.title("🏀 NBA Player Predictor")
    st.info("Run steps 1, 2, and 3 in the sidebar to build the system.")
    st.stop()


# Load models
points_model = load_model("points")
reb_model = load_model("rebounds")
ast_model = load_model("assists")

# Stop if models missing
if None in (points_model, reb_model, ast_model):
    st.title("🏀 NBA Player Predictor")
    st.warning("Models missing. Run Step 3 to train them.")
    st.stop()

feature_cols = get_feature_columns()


# ============================================================
#  MAIN UI — PLAYER SELECT
# ============================================================
st.title("🏀 NBA Player Predictor")

players = sorted(df["player_name"].unique())
player = st.selectbox("Choose a Player", players)

player_data = df[df["player_name"] == player].sort_values("GAME_DATE")
latest = player_data.tail(1).iloc[0]

X = latest[feature_cols].values.reshape(1, -1)

pred_pts = float(points_model.predict(X)[0])
pred_reb = float(reb_model.predict(X)[0])
pred_ast = float(ast_model.predict(X)[0])


# ============================================================
#  METRICS DISPLAY
# ============================================================
col1, col2, col3 = st.columns(3)
col1.metric("Projected Points", f"{pred_pts:.1f}")
col2.metric("Projected Rebounds", f"{pred_reb:.1f}")
col3.metric("Projected Assists", f"{pred_ast:.1f}")

st.markdown("---")

# ============================================================
#  RECENT GAME TREND CHART
# ============================================================
st.subheader("📈 Last 10 Games Trend")
chart_data = player_data.tail(10).set_index("GAME_DATE")[["points", "rebounds", "assists"]]
st.line_chart(chart_data)

st.markdown("---")

# ============================================================
#  FEATURE BREAKDOWN
# ============================================================
st.subheader("📊 Feature Breakdown Used for Prediction")
st.dataframe(latest[feature_cols], use_container_width=True)
