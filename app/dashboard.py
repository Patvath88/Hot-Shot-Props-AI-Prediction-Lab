# app/dashboard.py

import streamlit as st
import pandas as pd
import xgboost as xgb
from utils.features import get_feature_columns

st.set_page_config(page_title="NBA Player Predictor", layout="wide")

@st.cache_data
def load_dataset():
    return pd.read_csv("data/player_game_logs.csv")

@st.cache_resource
def load_model(target):
    model = xgb.XGBRegressor()
    model.load_model(f"models/{target}_xgb.json")
    return model


df = load_dataset()

points_model = load_model("points")
reb_model = load_model("rebounds")
ast_model = load_model("assists")

feature_cols = get_feature_columns()

st.title("🏀 NBA Player Stat Predictor")

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
