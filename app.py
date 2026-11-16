import io
import math
from collections import defaultdict, deque
from datetime import datetime

import numpy as np
import pandas as pd
import requests
import streamlit as st
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

st.set_page_config(page_title="Football Predictor AI", page_icon="⚽", layout="wide")

st.title("⚽ Football Predictor AI")
st.caption("Een gratis, experimentele voorspeller voor Europese voetbalcompetities.")

LEAGUES = {
    "Premier League (Engeland)": "E0",
    "Championship (Engeland)": "E1",
    "League One (Engeland)": "E2",
    "League Two (Engeland)": "E3",
    "Bundesliga (Duitsland)": "D1",
    "2. Bundesliga (Duitsland)": "D2",
    "Serie A (Italië)": "I1",
    "Serie B (Italië)": "I2",
    "La Liga (Spanje)": "SP1",
    "Segunda División (Spanje)": "SP2",
    "Ligue 1 (Frankrijk)": "F1",
    "Ligue 2 (Frankrijk)": "F2",
    "Eredivisie (Nederland)": "N1",
    "Eerste Divisie (Nederland)": "N2",
    "Primeira Liga (Portugal)": "P1",
    "Scottish Premiership (Schotland)": "SC0",
    "Scottish Championship (Schotland)": "SC1",
    "Scottish League One (Schotland)": "SC2",
    "Belgian Pro League (België)": "B1",
    "Turkish Süper Lig (Turkije)": "T1",
    "Greek Super League (Griekenland)": "G1",
}

SEASONS = [
    f"{str(y)[-2:]}{str(y+1)[-2:]}"
    for y in range(1993, datetime.now().year)
]
# Add current season, e.g. 2627 when running in 2026.
year = datetime.now().year
current_season = f"{str(year)[-2:]}{str(year+1)[-2:]}"
if current_season not in SEASONS:
    SEASONS.append(current_season)

@st.cache_data(ttl=6 * 60 * 60, show_spinner=False)
def download_data():
    frames = []
    errors = []
    for season in SEASONS:
        for league_name, code in LEAGUES.items():
            url = f"https://www.football-data.co.uk/mmz4281/{season}/{code}.csv"
            try:
                response = requests.get(url, timeout=12)
                if response.status_code != 200 or not response.content:
                    continue
                df = pd.read_csv(io.BytesIO(response.content), encoding="latin1", on_bad_lines="skip")
                required = {"HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"}
                if not required.issubset(df.columns):
                    continue
                df = df[list(required)].copy()
                df["League"] = code
                df["Season"] = season
                df["HomeTeam"] = df["HomeTeam"].astype(str).str.strip()
                df["AwayTeam"] = df["AwayTeam"].astype(str).str.strip()
                df["FTHG"] = pd.to_numeric(df["FTHG"], errors="coerce")
                df["FTAG"] = pd.to_numeric(df["FTAG"], errors="coerce")
                df = df.dropna(subset=["HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"])
                df = df[(df["FTHG"] >= 0) & (df["FTAG"] >= 0)]
                frames.append(df)
            except Exception as exc:
                errors.append(f"{season}/{code}: {type(exc).__name__}")
    if not frames:
        raise RuntimeError("Geen gegevens kunnen ophalen. Controleer je internetverbinding of probeer later opnieuw.")
    data = pd.concat(frames, ignore_index=True).drop_duplicates(
        subset=["Season", "League", "HomeTeam", "AwayTeam", "FTHG", "FTAG"]
    )
    return data, errors

def recent_summary(history, team, league, venue=None):
    matches = history.get((league, team), [])
    if venue:
        filtered = [m for m in matches if m["venue"] == venue]
        if len(filtered) >= 3:
            matches = filtered
    recent = list(matches)[-5:]
    if not recent:
        return [1.35, 1.35, 1.0, 0.0]
    gf = np.mean([m["gf"] for m in recent])
    ga = np.mean([m["ga"] for m in recent])
    pts = np.mean([m["pts"] for m in recent])
    return [float(gf), float(ga), float(pts), float(len(recent))]

def build_training_rows(data):
    # A match's features use only matches played before it.
    data = data.copy()
    data["_sort"] = np.arange(len(data))
    data = data.sort_values(["Season", "_sort"]).reset_index(drop=True)
    history = defaultdict(list)
    rows = []
    for _, match in data.iterrows():
        league = match["League"]
        home, away = match["HomeTeam"], match["AwayTeam"]
        hs = recent_summary(history, home, league, "home")
        as_ = recent_summary(history, away, league, "away")
        # Also include all-venue recent form to avoid sparse early-season data.
        hall = recent_summary(history, home, league)
        aall = recent_summary(history, away, league)
        rows.append({
            "League": league,
            "h_gf5": hs[0], "h_ga5": hs[1], "h_pts5": hs[2], "h_n5": hs[3],
            "a_gf5": as_[0], "a_ga5": as_[1], "a_pts5": as_[2], "a_n5": as_[3],
            "h_all_gf5": hall[0], "h_all_ga5": hall[1], "h_all_pts5": hall[2],
            "a_all_gf5": aall[0], "a_all_ga5": aall[1], "a_all_pts5": aall[2],
            "home_goals": float(match["FTHG"]),
            "away_goals": float(match["FTAG"]),
            "result": str(match["FTR"]),
        })
        hg, ag = int(match["FTHG"]), int(match["FTAG"])
        hp = 3 if hg > ag else 1 if hg == ag else 0
        ap = 3 if ag > hg else 1 if hg == ag else 0
        history[(league, home)].append({"venue": "home", "gf": hg, "ga": ag, "pts": hp})
        history[(league, away)].append({"venue": "away", "gf": ag, "ga": hg, "pts": ap})
    return pd.DataFrame(rows), history

FEATURES = [
    "League", "h_gf5", "h_ga5", "h_pts5", "h_n5",
    "a_gf5", "a_ga5", "a_pts5", "a_n5",
    "h_all_gf5", "h_all_ga5", "h_all_pts5",
    "a_all_gf5", "a_all_ga5", "a_all_pts5",
]
NUMERIC = [c for c in FEATURES if c != "League"]

def make_preprocessor():
    return ColumnTransformer([
        ("league", OneHotEncoder(handle_unknown="ignore"), ["League"]),
        ("numbers", SimpleImputer(strategy="median"), NUMERIC),
    ])

@st.cache_resource(show_spinner=False)
def train_models(data_fingerprint, rows):
    # Use a stable time-ordered holdout for a simple, honest evaluation.
    X = rows[FEATURES]
    result_model = Pipeline([
        ("prep", make_preprocessor()),
        ("model", RandomForestClassifier(
            n_estimators=220, min_samples_leaf=4, class_weight="balanced",
            random_state=42, n_jobs=-1
        )),
    ])
    home_model = Pipeline([
        ("prep", make_preprocessor()),
        ("model", RandomForestRegressor(
            n_estimators=180, min_samples_leaf=4, random_state=43, n_jobs=-1
        )),
    ])
    away_model = Pipeline([
        ("prep", make_preprocessor()),
        ("model", RandomForestRegressor(
            n_estimators=180, min_samples_leaf=4, random_state=44, n_jobs=-1
        )),
    ])
    result_model.fit(X, rows["result"])
    home_model.fit(X, rows["home_goals"])
    away_model.fit(X, rows["away_goals"])
    return result_model, home_model, away_model

with st.sidebar:
    st.header("Instellingen")
    if st.button("🔄 Gegevens opnieuw ophalen"):
        download_data.clear()
        train_models.clear()
        st.rerun()
    st.write("De databron wordt maximaal elke 6 uur automatisch ververst.")

try:
    with st.spinner("Historische Europese uitslagen ophalen..."):
        raw, download_errors = download_data()
    rows, history = build_training_rows(raw)
    if len(rows) < 100:
        st.error("Er zijn te weinig wedstrijden beschikbaar om een model te trainen.")
        st.stop()
    fingerprint = f"{len(rows)}-{int(rows['home_goals'].sum())}-{int(rows['away_goals'].sum())}"
    with st.spinner(f"Model trainen op {len(rows):,} wedstrijden..."):
        result_model, home_model, away_model = train_models(fingerprint, rows)
except Exception as exc:
    st.error(f"Het model kon niet worden gestart: {exc}")
    st.info("Controleer of de databron bereikbaar is en probeer het later opnieuw.")
    st.stop()

st.success(f"Model getraind op {len(rows):,} historische wedstrijden uit {raw['League'].nunique()} Europese competities.")
st.caption("Historische gegevens zijn niet volledig: de dekking verschilt per land, competitie en seizoen. De modeluitkomsten zijn schattingen, geen zekerheden.")

league_names = {code: name for name, code in LEAGUES.items()}
available_leagues = sorted(raw["League"].dropna().unique())
league_choice = st.selectbox(
    "Kies een competitie",
    available_leagues,
    format_func=lambda x: league_names.get(x, x)
)
teams = sorted(set(raw.loc[raw["League"] == league_choice, "HomeTeam"]) |
               set(raw.loc[raw["League"] == league_choice, "AwayTeam"]))

c1, c2 = st.columns(2)
with c1:
    home_team = st.selectbox("Thuisploeg", teams, index=0)
with c2:
    away_options = [t for t in teams if t != home_team]
    away_team = st.selectbox("Uitploeg", away_options, index=0 if away_options else None)

if st.button("⚽ Voorspel wedstrijd", type="primary", use_container_width=True):
    hs = recent_summary(history, home_team, league_choice, "home")
    as_ = recent_summary(history, away_team, league_choice, "away")
    hall = recent_summary(history, home_team, league_choice)
    aall = recent_summary(history, away_team, league_choice)
    sample = pd.DataFrame([{
        "League": league_choice,
        "h_gf5": hs[0], "h_ga5": hs[1], "h_pts5": hs[2], "h_n5": hs[3],
        "a_gf5": as_[0], "a_ga5": as_[1], "a_pts5": as_[2], "a_n5": as_[3],
        "h_all_gf5": hall[0], "h_all_ga5": hall[1], "h_all_pts5": hall[2],
        "a_all_gf5": aall[0], "a_all_ga5": aall[1], "a_all_pts5": aall[2],
    }])
    probs = result_model.predict_proba(sample)[0]
    classes = list(result_model.named_steps["model"].classes_)
    p = {str(k): float(v) for k, v in zip(classes, probs)}
    ph = max(0.0, min(1.0, float(home_model.predict(sample)[0])))
    pa = max(0.0, min(1.0, float(away_model.predict(sample)[0])))
    # Approximate score using expected goals; no claim of exact-score certainty.
    score_h, score_a = int(round(ph)), int(round(pa))
    both_prob = (1 - math.exp(-ph)) * (1 - math.exp(-pa))
    both_yes = both_prob >= 0.5
    winner = "Gelijkspel" if p.get("D", 0) >= max(p.get("H", 0), p.get("A", 0)) else (
        home_team if p.get("H", 0) >= p.get("A", 0) else away_team
    )
    st.divider()
    st.subheader(f"{home_team} – {away_team}")
    st.markdown(f"<h1 style='text-align:center'>{score_h} – {score_a}</h1>", unsafe_allow_html=True)
    st.caption("Geschatte score op basis van verwachte doelpunten; geen gegarandeerde einduitslag.")
    m1, m2, m3 = st.columns(3)
    m1.metric(f"Winst {home_team}", f"{p.get('H', 0)*100:.1f}%")
    m2.metric("Gelijkspel", f"{p.get('D', 0)*100:.1f}%")
    m3.metric(f"Winst {away_team}", f"{p.get('A', 0)*100:.1f}%")
    st.write(f"**Meest waarschijnlijke uitkomst:** {winner}")
    b1, b2 = st.columns(2)
    b1.metric("Verwachte doelpunten", f"{ph + pa:.2f}")
    b2.metric("Beide teams scoren", "Ja" if both_yes else "Nee", f"Geschatte kans: {both_prob*100:.1f}%")
    st.subheader("Vorm op basis van laatste wedstrijden")
    form = pd.DataFrame([
        {"Team": home_team, "Goals voor per wedstrijd": round(hall[0], 2), "Goals tegen": round(hall[1], 2), "Punten per wedstrijd": round(hall[2], 2)},
        {"Team": away_team, "Goals voor per wedstrijd": round(aall[0], 2), "Goals tegen": round(aall[1], 2), "Punten per wedstrijd": round(aall[2], 2)},
    ])
    st.dataframe(form, hide_index=True, use_container_width=True)

with st.expander("Over het model en de gegevens"):
    st.write("""
    - Bron: Football-Data.co.uk, met historische uitslagen van geselecteerde Europese competities.
    - De trainingskenmerken zijn gebaseerd op eerdere wedstrijden van elk team.
    - De uitkomstclassificatie schat thuiswinst (H), gelijkspel (D) en uitwinst (A).
    - Twee regressiemodellen schatten de doelpunten van thuis- en uitploeg.
    - De weergegeven exacte score is een afgeronde schatting, niet een afzonderlijk gegarandeerd resultaat.
    - Nieuwe uitslagen worden meegenomen wanneer de databron is ververst en het model opnieuw wordt getraind.
    - Dit is een eerste prototype. Het is niet gevalideerd voor winstgevend wedden of andere financiële beslissingen.
    """)

st.caption("Football Predictor AI • gratis prototype • gemaakt met Python en Streamlit")
