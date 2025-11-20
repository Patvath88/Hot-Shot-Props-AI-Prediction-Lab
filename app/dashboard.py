# app/dashboard.py

import sys
import os
import subprocess
import streamlit as st
import pandas as pd
import xgboost as xgb

APP_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(APP_DIR, ".."))
sys.path.insert(0, ROOT_DIR)

from utils.features import get_feature_columns


def run_pipeline_step(label, script):
    st.write(f"### 🔧 {label}")

    process = subprocess.Popen(
        [sys.executable, os.path.join(ROOT_DIR, script)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    out, err = process.communicate()
    if out: st.success(out.replace("\n", "  \n"))
    if err: st.error(err.replace("\n", "  \n"))


st.sidebar.header("⚙️ Unified Pipeline")

if st.sidebar.button("🔄 Run FULL Pipeline"):
    run_pipeline_step("1️⃣ Scraping Logs", "scrapers/scrape_basic_logs_fast.py")
    run_pipeline_step("2️⃣ Building Dataset", "scrapers/build_dataset.py")
    run_pipeline_step("3️⃣ Training Models", "train/train_all.py")
    st.sidebar.success("🎉 Full pipeline done!")
    st.rerun()

st.sidebar.markdown("---")

if st.sidebar.button("1️⃣ Scrape Logs"):
    run_pipeline_step("Scraping Logs", "scrapers/scrape_basic_logs_fast.py")

if st.sidebar.button("2️⃣ Build Dataset"):
    run_pipeline_step("Build Dataset", "scrapers/build_dataset.py")

if st.sidebar.button("3️⃣ Train Models"):
    run_pipeline_step("Train Models", "train/train_all.py")

st.sidebar.markdown("---")


@st.cache_data
def load_dataset():
    path = os.path.join(ROOT_DIR, "data/player_game_logs.csv")
    if not os.path.exists(path): return pd.DataFrame()
    return pd.read_csv(path)


@st.cache_resource
def load_model(name):
    model_path = os.path.join(ROOT_DIR, "models", f"{name}_xgb.json")
    if not os.path.exists(model_path): return None
    model = xgb.XGBRegressor()
    model.load_model(model_path)
    return model


df = load_dataset()
if df.empty:
    st.title("🏀 NBA Player Predictor")
    st.info("Run the full pipeline first.")
    st.stop()

points_model = load_model("points")
reb_model = load_model("rebounds")
ast_model = load_model("assists")

if None in (points_model, reb_model, ast_model):
    st.warning("Models missing — train them using Step 3.")
    st.stop()

feature_cols = get_feature_columns()

st.title("🏀 NBA Player Stat Predictor")

players = sorted(df["player_name"].unique())
player = st.selectbox("Select Player", players)

pdf = df[df["player_name"] == player].sort_values("GAME_DATE")
latest = pdf.tail(1).iloc[0]

X = latest[feature_cols].values.reshape(1, -1)

pred_pts = float(points_model.predict(X)[0])
pred_reb = float(reb_model.predict(X)[0])
pred_ast = float(ast_model.predict(X)[0])

c1, c2, c3 = st.columns(3)
c1.metric("Projected Points", f"{pred_pts:.1f}")
c2.metric("Projected Rebounds", f"{pred_reb:.1f}")
c3.metric("Projected Assists", f"{pred_ast:.1f}")

st.subheader("📈 Last 10 Games")
st.line_chart(pdf.tail(10).set_index("GAME_DATE")[["points", "rebounds", "assists"]])

st.subheader("🔍 Features Used")
st.dataframe(latest[feature_cols], use_container_width=True)