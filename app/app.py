# app/app.py
import os
import math
import pandas as pd
import streamlit as st

from utils.features import get_feature_columns
from utils.io_utils import load_model


# ------------------ CONFIG / STYLING ------------------ #

st.set_page_config(
    page_title="NBA Prop Edge Engine",
    page_icon="🏀",
    layout="wide",
)

st.markdown("""
<style>
/* Dark premium theme */
.stApp {
    background: radial-gradient(circle at top, #19233d 0, #050712 55%, #020309 100%);
    color: #ffffff;
    font-family: "Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}
.block-container {
    padding-top: 1.5rem;
    padding-bottom: 2rem;
}
.big-number {
    font-size: 2.6rem;
    font-weight: 800;
}
.sub-label {
    font-size: 0.8rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #aeb3c7;
}
.stat-card {
    padding: 1rem 1.25rem;
    border-radius: 1rem;
    border: 1px solid rgba(255,255,255,0.05);
    background: linear-gradient(145deg, rgba(255,255,255,0.04), rgba(0,0,0,0.35));
    backdrop-filter: blur(12px);
}
.edge-positive {
    color: #4ade80;
}
.edge-negative {
    color: #fb7185;
}
.tag {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    padding: 0.25rem 0.6rem;
    border-radius: 999px;
    font-size: 0.75rem;
    background: rgba(15, 118, 110, 0.18);
    color: #5eead4;
    border: 1px solid rgba(45, 212, 191, 0.5);
}
</style>
""", unsafe_allow_html=True)


# ------------------ HELPERS ------------------ #

def american_to_implied_prob(odds: float) -> float:
    if odds > 0:
        return 100.0 / (odds + 100.0)
    return -odds / (-odds + 100.0)


def implied_prob_to_american(p: float) -> float:
    if p <= 0 or p >= 1:
        return None
    if p > 0.5:
        # favorite
        return -int(round(p / (1 - p) * 100))
    else:
        # dog
        return int(round((1 - p) / p * 100))


@st.cache_data
def load_upcoming_props(path: str = "data/upcoming_props.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


@st.cache_resource
def load_models():
    models = {}
    for prop_type, fname in [
        ("points", "models/points_xgb.json"),
        ("rebounds", "models/rebounds_xgb.json"),
        ("assists", "models/assists_xgb.json"),
    ]:
        if os.path.exists(fname):
            models[prop_type] = load_model(fname)
    return models


def predict_row(model, row_features):
    return float(model.predict(row_features.values.reshape(1, -1))[0])


# ------------------ MAIN APP ------------------ #

st.markdown("### 🏀 NBA Prop Edge Engine")

col_title, col_tag = st.columns([0.7, 0.3])
with col_title:
    st.markdown(
        "<span style='font-size:2rem;font-weight:800;'>Tonight's AI Projections</span>",
        unsafe_allow_html=True
    )
with col_tag:
    st.markdown(
        "<div style='text-align:right;margin-top:0.5rem;'>"
        "<span class='tag'>XGBoost • Feature Engineered</span>"
        "</div>",
        unsafe_allow_html=True
    )

models = load_models()
upcoming_df = load_upcoming_props()

if upcoming_df.empty or not models:
    st.warning("Models or upcoming props not found. Make sure models are trained and `data/upcoming_props.csv` exists.")
    st.stop()

feature_cols = get_feature_columns()

# Sidebar controls
st.sidebar.header("Filter Slate")

slate_date = st.sidebar.selectbox(
    "Slate Date",
    sorted(upcoming_df["slate_date"].unique()) if "slate_date" in upcoming_df.columns else ["Today"],
)

prop_types_available = sorted(upcoming_df["prop_type"].unique())
selected_prop_type = st.sidebar.selectbox(
    "Prop Type",
    prop_types_available,
    index=prop_types_available.index("points") if "points" in prop_types_available else 0
)

filtered = upcoming_df.copy()
if "slate_date" in upcoming_df.columns and slate_date != "Today":
    filtered = filtered[filtered["slate_date"] == slate_date]

filtered = filtered[filtered["prop_type"] == selected_prop_type]

players = sorted(filtered["player_name"].unique())
selected_player = st.sidebar.selectbox("Player", players)

player_rows = filtered[filtered["player_name"] == selected_player]

if player_rows.empty:
    st.warning("No props available for this player/prop type.")
    st.stop()

# Assume one row per player/prop for slate
row = player_rows.iloc[0]

# Model for this prop type
if selected_prop_type not in models:
    st.error(f"No trained model found for prop type: {selected_prop_type}")
    st.stop()

model = models[selected_prop_type]

# Build feature vector
missing_cols = [c for c in feature_cols if c not in row.index]
if missing_cols:
    st.error(f"Missing feature columns in upcoming_props.csv: {missing_cols}")
    st.stop()

X_row = row[feature_cols].astype(float)

projection = predict_row(model, X_row)
line = float(row["line"])
american_odds = float(row.get("american_odds", -110))

# Basic edge: model prob for over vs implied prob
# Assuming normal-ish error, quick and dirty:
sigma = max(0.01, row.get("proj_std", 3.0))  # fallback
z = (projection - line) / sigma
# Convert z to probability Over using normal CDF approximation:
# For simplicity a logistic approximation:
from math import exp
prob_over_model = 1.0 / (1.0 + exp(-1.6 * z))

prob_over_implied = american_to_implied_prob(american_odds)
edge = prob_over_model - prob_over_implied
edge_pct = edge * 100

model_ao = implied_prob_to_american(prob_over_model)

# ------------------ TOP SECTION: HERO STATS ------------------ #

top_left, top_mid, top_right = st.columns([1.4, 1.1, 1.1])

with top_left:
    st.markdown("##### Player / Market")
    st.markdown(
        f"""
        <div class="stat-card">
            <div class="sub-label">Player</div>
            <div class="big-number">{row['player_name']}</div>
            <div style="margin-top:0.35rem;opacity:0.85;">
                {row['team']} vs {row['opp_team']}
            </div>
            <div style="margin-top:0.75rem;" class="sub-label">Market</div>
            <div style="font-size:1.1rem;font-weight:600;margin-top:0.1rem;">
                {selected_prop_type.capitalize()} {line} ({american_odds:+.0f})
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with top_mid:
    st.markdown("##### Model Projection")
    st.markdown(
        f"""
        <div class="stat-card">
            <div class="sub-label">Projected {selected_prop_type}</div>
            <div class="big-number">{projection:.1f}</div>
            <div style="margin-top:0.75rem;" class="sub-label">Model Odds (Over)</div>
            <div style="font-size:1.4rem;font-weight:700;margin-top:0.15rem;">
                {model_ao:+.0f} 
                <span style="font-size:0.9rem;opacity:0.8;">({prob_over_model*100:.1f}%</span>
                <span style="font-size:0.8rem;opacity:0.8;"> implied)</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with top_right:
    edge_class = "edge-positive" if edge_pct >= 0 else "edge-negative"
    edge_label = "Model Edge (Over)"
    st.markdown("##### Edge vs Book")
    st.markdown(
        f"""
        <div class="stat-card">
            <div class="sub-label">{edge_label}</div>
            <div class="big-number {edge_class}">
                {edge_pct:+.1f}%
            </div>
            <div style="margin-top:0.75rem;" class="sub-label">Book Implied Over</div>
            <div style="font-size:1.1rem;font-weight:600;margin-top:0.1rem;">
                {prob_over_implied*100:.1f}% ({american_odds:+.0f})
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("---")

# ------------------ MIDDLE: CONTEXT & FORM ------------------ #

st.markdown("#### Context & Recent Form")

c1, c2, c3, c4 = st.columns(4)

def display_stat_card(col, label, value, sub=None):
    with col:
        st.markdown(
            f"""
            <div class="stat-card">
                <div class="sub-label">{label}</div>
                <div class="big-number" style="font-size:1.8rem;">{value}</div>
                <div style="margin-top:0.35rem;opacity:0.8;font-size:0.8rem;">{sub or ""}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

display_stat_card(
    c1,
    "Season Avg",
    f"{row.get(f'{selected_prop_type}_avg_season', float('nan')):.1f}",
    f"{selected_prop_type.capitalize()} per game"
)
display_stat_card(
    c2,
    "Last 5 Avg",
    f"{row.get(f'{selected_prop_type}_rolling_5', float('nan')):.1f}",
    "Last 5 games"
)
display_stat_card(
    c3,
    "Minutes (L5)",
    f"{row.get('minutes_rolling', float('nan')):.1f}",
    "Avg minutes last 5"
)
display_stat_card(
    c4,
    "Form Score",
    f"{row.get('form_score', float('nan')):.1f}",
    "Composite usage/production"
)

# ------------------ TABLE: ALL EDGES ON SLATE ------------------ #

st.markdown("#### Full Slate Edges")

table_df = []
for _, r in filtered.iterrows():
    prop_type = r["prop_type"]
    if prop_type not in models:
        continue

    # Make sure all features exist
    if any(c not in r.index for c in feature_cols):
        continue

    Xr = r[feature_cols].astype(float)
    proj = predict_row(models[prop_type], Xr)
    line_r = float(r["line"])
    odds_r = float(r.get("american_odds", -110))

    sigma_r = max(0.01, r.get("proj_std", 3.0))
    z_r = (proj - line_r) / sigma_r
    prob_over_r = 1.0 / (1.0 + math.exp(-1.6 * z_r))
    prob_imp_r = american_to_implied_prob(odds_r)
    edge_pct_r = (prob_over_r - prob_imp_r) * 100
    model_ao_r = implied_prob_to_american(prob_over_r)

    table_df.append({
        "Player": r["player_name"],
        "Team": r["team"],
        "Opp": r["opp_team"],
        "Prop": prop_type,
        "Line": line_r,
        "Book Odds": odds_r,
        "Model Proj": round(proj, 2),
        "Model Over %": round(prob_over_r * 100, 1),
        "Model Odds": model_ao_r,
        "Edge % (Over)": round(edge_pct_r, 1),
    })

if table_df:
    table_df = pd.DataFrame(table_df)
    # sort by edge
    table_df = table_df.sort_values("Edge % (Over)", ascending=False)
    st.dataframe(
        table_df,
        use_container_width=True,
        hide_index=True,
    )
else:
    st.info("No props available with matching features and models.")
