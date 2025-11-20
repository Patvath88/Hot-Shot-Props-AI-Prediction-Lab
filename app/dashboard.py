# app/dashboard.py

import streamlit as st
import pandas as pd
import xgboost as xgb
import numpy as np

from utils.features import get_feature_columns

st.set_page_config(page_title="NBA Player Stat Predictor", layout="wide")

# ---------------- Load Data & Models ---------------- #

@st.cache_data
def load_logs():
    return pd.read_csv("data/player_game_logs.csv")

@st.cache_resource
def load_model(target):
    model = xgb.XGBRegressor()
    model.load_model(f"models/{target}_xgb.json")
    return model


df = load_logs()
points_model = load_model("points")
reb_model = load_model("rebounds")
ast_model = load_model("assists")

feature_cols = get_feature_columns()


# ---------------- Sidebar UI ---------------- #

st.sidebar.header("Player Selection")

players = sorted(df["player_name"].dropna().unique())
player = st.sidebar.selectbox("Player", players)

player_df = df[df["player_name"] == player].sort_values("GAME_DATE")


# ---------------- Prediction Logic ---------------- #

latest = player_df.tail(1).iloc[0]

X = latest[feature_cols].values.reshape(1, -1)

pred_points = float(points_model.predict(X)[0])
pred_rebounds = float(reb_model.predict(X)[0])
pred_assists = float(ast_model.predict(X)[0])

# Form score
form_score = float(latest.get("form_score", np.nan))

# Difficulty (lower opp def rating = better)
difficulty = float(latest.get("DEF_RATING", np.nan))


# ---------------- Dashboard Layout ---------------- #

st.title(f"🏀 {player} — Next Game Projections")

col1, col2, col3 = st.columns(3)

col1.metric("Projected POINTS", f"{pred_points:.1f}")
col2.metric("Projected REBOUNDS", f"{pred_rebounds:.1f}")
col3.metric("Projected ASSISTS", f"{pred_assists:.1f}")

st.markdown("----")

colA, colB = st.columns([1, 1])

with colA:
    st.subheader("📈 Recent Performance (Last 10 Games)")
    recent = player_df.tail(10)[["GAME_DATE", "points", "rebounds", "assists"]]
    st.line_chart(recent.set_index("GAME_DATE"))

with colB:
    st.subheader("🔥 Player Form")
    st.metric("Form Score", f"{form_score:.2f}")
    st.metric("Opponent Difficulty (DEF Rating)", f"{difficulty:.1f}")

st.markdown("----")

# ---------------- Advanced Context ---------------- #

st.subheader("📊 Detailed Feature Table")
st.dataframe(
    latest[feature_cols + ["points", "rebounds", "assists"]],
    hide_index=True
)
