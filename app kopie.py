import io
import math
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import numpy as np
import pandas as pd
import requests
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import log_loss

st.set_page_config(page_title="Bundesliga Predictor AI", page_icon="⚽", layout="wide")

st.title("⚽ Bundesliga Predictor AI")
st.caption("Een gratis, experimentele voorspeller voor de Duitse Bundesliga.")

LEAGUE_CODE = "D1"          # Bundesliga bij football-data.co.uk
FIRST_SEASON_YEAR = 1993    # eerste seizoen met data: 1993/94
MAX_GOALS = 8               # grootste aantal goals per team in de scorematrix
FORM_GAMES = 5              # korte termijn vorm
LONG_GAMES = 20             # lange termijn vorm
MIN_GAMES = 3               # minimaal aantal eerdere wedstrijden per team om mee te trainen
OUTCOMES = ["H", "D", "A"]


# ---------------------------------------------------------------- data ophalen
def season_codes():
    """Seizoenscodes zoals '9394'. Een seizoen start in augustus."""
    now = datetime.now()
    last_start = now.year if now.month >= 7 else now.year - 1
    return [f"{str(y)[-2:]}{str(y + 1)[-2:]}" for y in range(FIRST_SEASON_YEAR, last_start + 1)]


def fetch_season(season):
    url = f"https://www.football-data.co.uk/mmz4281/{season}/{LEAGUE_CODE}.csv"
    needed = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"]
    try:
        response = requests.get(url, timeout=15)
        if response.status_code != 200 or not response.content:
            return None
        df = pd.read_csv(io.BytesIO(response.content), encoding="latin1", on_bad_lines="skip")
        if not set(needed).issubset(df.columns):
            return None
        df = df[needed].copy()
        df["Season"] = season
        return df
    except Exception:
        return None


@st.cache_data(ttl=6 * 60 * 60, show_spinner=False)
def download_data():
    # Alle seizoenen tegelijk ophalen (veel sneller dan één voor één).
    with ThreadPoolExecutor(max_workers=8) as pool:
        frames = [f for f in pool.map(fetch_season, season_codes()) if f is not None]
    if not frames:
        raise RuntimeError("Geen gegevens kunnen ophalen. Controleer je internetverbinding en probeer het later opnieuw.")

    data = pd.concat(frames, ignore_index=True)
    data["HomeTeam"] = data["HomeTeam"].astype(str).str.strip()
    data["AwayTeam"] = data["AwayTeam"].astype(str).str.strip()
    data["FTHG"] = pd.to_numeric(data["FTHG"], errors="coerce")
    data["FTAG"] = pd.to_numeric(data["FTAG"], errors="coerce")
    data["Date"] = pd.to_datetime(data["Date"], dayfirst=True, format="mixed", errors="coerce")
    data = data.dropna(subset=["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"])
    data = data[(data["FTHG"] >= 0) & (data["FTAG"] >= 0)]
    data["FTHG"] = data["FTHG"].astype(int)
    data["FTAG"] = data["FTAG"].astype(int)
    # Echte chronologische volgorde (op datum, niet op seizoenstekst).
    data = data.sort_values("Date", kind="stable").reset_index(drop=True)
    data = data.drop_duplicates(subset=["Date", "HomeTeam", "AwayTeam"]).reset_index(drop=True)
    return data


# ---------------------------------------------------------------- kenmerken
def summarize(games, venue=None, n=FORM_GAMES):
    """Gemiddelde goals voor, goals tegen en punten over de laatste n wedstrijden."""
    games = list(games)
    if venue:
        same_venue = [g for g in games if g[0] == venue]
        if len(same_venue) >= MIN_GAMES:
            games = same_venue
    games = games[-n:]
    if not games:
        return 1.5, 1.5, 1.35  # gemiddelde waarden als er nog geen geschiedenis is
    return (
        float(np.mean([g[1] for g in games])),
        float(np.mean([g[2] for g in games])),
        float(np.mean([g[3] for g in games])),
    )


TEAM_FEATURES = ["gf5", "ga5", "pts5", "gf20", "ga20", "pts20", "vgf5", "vga5", "n"]
FEATURES = [f"h_{f}" for f in TEAM_FEATURES] + [f"a_{f}" for f in TEAM_FEATURES]


def team_features(history, team, venue):
    games = history[team]
    s5 = summarize(games, None, FORM_GAMES)
    s20 = summarize(games, None, LONG_GAMES)
    sv = summarize(games, venue, FORM_GAMES)
    return [s5[0], s5[1], s5[2], s20[0], s20[1], s20[2], sv[0], sv[1], float(len(games))]


def new_history():
    return defaultdict(lambda: deque(maxlen=LONG_GAMES))


def build_rows(data):
    """Kenmerken van elke wedstrijd gebruiken alleen wedstrijden die ervoor zijn gespeeld."""
    history = new_history()
    rows = []
    for m in data.itertuples(index=False):
        h = team_features(history, m.HomeTeam, "home")
        a = team_features(history, m.AwayTeam, "away")
        row = dict(zip(FEATURES, h + a))
        row.update({
            "Date": m.Date, "Season": m.Season,
            "home_goals": m.FTHG, "away_goals": m.FTAG,
            "result": "H" if m.FTHG > m.FTAG else "D" if m.FTHG == m.FTAG else "A",
        })
        rows.append(row)
        hp = 3 if m.FTHG > m.FTAG else 1 if m.FTHG == m.FTAG else 0
        ap = 3 if m.FTAG > m.FTHG else 1 if m.FTHG == m.FTAG else 0
        history[m.HomeTeam].append(("home", m.FTHG, m.FTAG, hp))
        history[m.AwayTeam].append(("away", m.FTAG, m.FTHG, ap))
    return pd.DataFrame(rows), history


# ---------------------------------------------------------------- Poisson
def poisson_pmf(lam):
    return np.array([math.exp(-lam) * lam ** k / math.factorial(k) for k in range(MAX_GOALS + 1)])


def score_matrix(lam_home, lam_away):
    """Kans op elke uitslag (rij = thuisgoals, kolom = uitgoals)."""
    m = np.outer(poisson_pmf(lam_home), poisson_pmf(lam_away))
    return m / m.sum()


def outcome_probs(m):
    home = float(np.tril(m, -1).sum())
    draw = float(np.trace(m))
    away = float(np.triu(m, 1).sum())
    return {"H": home, "D": draw, "A": away}


def make_model(seed):
    return RandomForestRegressor(
        n_estimators=200, min_samples_leaf=10, max_features=0.5,
        random_state=seed, n_jobs=-1,
    )


def predict_lambdas(models, sample):
    lam_h = float(np.clip(models["home"].predict(sample)[0], 0.2, 5.0))
    lam_a = float(np.clip(models["away"].predict(sample)[0], 0.2, 5.0))
    return lam_h, lam_a


def evaluate(rows):
    """Test eerlijk: train op het verleden, test op het laatste volledige seizoen."""
    season_start = rows.groupby("Season")["Date"].min().sort_values()
    season_size = rows.groupby("Season").size()
    complete = [s for s in season_start.index if season_size[s] >= 300]
    if not complete:
        return None
    test_season = complete[-1]
    cutoff = season_start[test_season]
    train = rows[(rows["Date"] < cutoff) & (rows["h_n"] >= MIN_GAMES) & (rows["a_n"] >= MIN_GAMES)]
    test = rows[(rows["Season"] == test_season) & (rows["h_n"] >= MIN_GAMES) & (rows["a_n"] >= MIN_GAMES)]
    if len(train) < 500 or len(test) < 100:
        return None
    models = {
        "home": make_model(43).fit(train[FEATURES], train["home_goals"]),
        "away": make_model(44).fit(train[FEATURES], train["away_goals"]),
    }
    probs = []
    for _, r in test.iterrows():
        lh, la = predict_lambdas(models, pd.DataFrame([r[FEATURES]]))
        p = outcome_probs(score_matrix(lh, la))
        probs.append([p[o] for o in OUTCOMES])
    probs = np.array(probs)
    predicted = [OUTCOMES[i] for i in probs.argmax(axis=1)]
    accuracy = float(np.mean(np.array(predicted) == test["result"].values))
    baseline_class = train["result"].value_counts().idxmax()
    baseline = float(np.mean(test["result"].values == baseline_class))
    loss = float(log_loss(test["result"], probs, labels=OUTCOMES))
    return {
        "season": test_season, "n": len(test),
        "accuracy": accuracy, "baseline": baseline, "log_loss": loss,
    }


@st.cache_resource(show_spinner=False)
def build_everything(_raw, fingerprint):
    rows, history = build_rows(_raw)
    train = rows[(rows["h_n"] >= MIN_GAMES) & (rows["a_n"] >= MIN_GAMES)]
    models = {
        "home": make_model(43).fit(train[FEATURES], train["home_goals"]),
        "away": make_model(44).fit(train[FEATURES], train["away_goals"]),
    }
    metrics = evaluate(rows)
    return rows, history, models, metrics


# ---------------------------------------------------------------- app
with st.sidebar:
    st.header("Instellingen")
    if st.button("🔄 Gegevens opnieuw ophalen"):
        download_data.clear()
        build_everything.clear()
        st.rerun()
    st.write("De databron wordt maximaal elke 6 uur automatisch ververst.")

try:
    with st.spinner("Bundesliga-uitslagen ophalen..."):
        raw = download_data()
    fingerprint = f"{len(raw)}-{raw['Date'].max():%Y%m%d}"
    with st.spinner(f"Model trainen op {len(raw):,} wedstrijden..."):
        rows, history, models, metrics = build_everything(raw, fingerprint)
except Exception as exc:
    st.error(f"Het model kon niet worden gestart: {exc}")
    st.info("Controleer of de databron bereikbaar is en probeer het later opnieuw.")
    st.stop()

if len(rows) < 300:
    st.error("Er zijn te weinig wedstrijden beschikbaar om een model te trainen.")
    st.stop()

st.success(
    f"Model getraind op {len(rows):,} Bundesliga-wedstrijden "
    f"(laatste uitslag: {raw['Date'].max():%d-%m-%Y})."
)
st.caption("De uitkomsten zijn schattingen, geen zekerheden.")

# Alleen teams uit de laatste twee seizoenen laten zien.
season_order = raw.groupby("Season")["Date"].min().sort_values().index.tolist()
recent = raw[raw["Season"].isin(season_order[-2:])]
teams = sorted(set(recent["HomeTeam"]) | set(recent["AwayTeam"]))

c1, c2 = st.columns(2)
with c1:
    home_team = st.selectbox("Thuisploeg", teams, index=0)
with c2:
    away_options = [t for t in teams if t != home_team]
    away_team = st.selectbox("Uitploeg", away_options, index=0)

if st.button("⚽ Voorspel wedstrijd", type="primary"):
    sample = pd.DataFrame([dict(zip(
        FEATURES,
        team_features(history, home_team, "home") + team_features(history, away_team, "away"),
    ))])
    lam_h, lam_a = predict_lambdas(models, sample)

    # Alles komt uit dezelfde scorematrix, dus score en kansen spreken elkaar niet tegen.
    matrix = score_matrix(lam_h, lam_a)
    p = outcome_probs(matrix)
    best_h, best_a = np.unravel_index(matrix.argmax(), matrix.shape)
    btts = float(matrix[1:, 1:].sum())
    over25 = float(sum(matrix[i, j] for i in range(MAX_GOALS + 1) for j in range(MAX_GOALS + 1) if i + j >= 3))

    winner = {"H": home_team, "D": "Gelijkspel", "A": away_team}[max(p, key=p.get)]

    st.divider()
    st.subheader(f"{home_team} – {away_team}")
    st.markdown(f"<h1 style='text-align:center'>{best_h} – {best_a}</h1>", unsafe_allow_html=True)
    st.caption("Meest waarschijnlijke uitslag. Geen enkele uitslag is zeker.")

    m1, m2, m3 = st.columns(3)
    m1.metric(f"Winst {home_team}", f"{p['H'] * 100:.1f}%")
    m2.metric("Gelijkspel", f"{p['D'] * 100:.1f}%")
    m3.metric(f"Winst {away_team}", f"{p['A'] * 100:.1f}%")
    st.write(f"**Meest waarschijnlijke uitkomst:** {winner}")

    b1, b2, b3 = st.columns(3)
    b1.metric("Verwachte doelpunten", f"{lam_h + lam_a:.2f}")
    b2.metric("Beide teams scoren", f"{btts * 100:.1f}%")
    b3.metric("Meer dan 2,5 goals", f"{over25 * 100:.1f}%")

    st.subheader("Top 5 meest waarschijnlijke uitslagen")
    flat = sorted(
        ((matrix[i, j], i, j) for i in range(MAX_GOALS + 1) for j in range(MAX_GOALS + 1)),
        reverse=True,
    )[:5]
    st.dataframe(
        pd.DataFrame({"Uitslag": [f"{i} – {j}" for _, i, j in flat],
                      "Kans": [f"{pr * 100:.1f}%" for pr, _, _ in flat]}),
        hide_index=True,
    )

    st.subheader("Vorm (laatste 5 wedstrijden)")
    hs, as_ = summarize(history[home_team]), summarize(history[away_team])
    st.dataframe(
        pd.DataFrame([
            {"Team": home_team, "Goals voor": round(hs[0], 2), "Goals tegen": round(hs[1], 2), "Punten per wedstrijd": round(hs[2], 2)},
            {"Team": away_team, "Goals voor": round(as_[0], 2), "Goals tegen": round(as_[1], 2), "Punten per wedstrijd": round(as_[2], 2)},
        ]),
        hide_index=True,
    )

with st.expander("Hoe goed is het model?"):
    if metrics:
        st.write(
            f"Getest op het seizoen **{metrics['season']}** ({metrics['n']} wedstrijden), "
            f"met een model dat alleen wedstrijden van vóór dat seizoen kende."
        )
        t1, t2, t3 = st.columns(3)
        t1.metric("Juiste uitkomst (W/G/V)", f"{metrics['accuracy'] * 100:.1f}%")
        t2.metric("Altijd de vaakste uitkomst raden", f"{metrics['baseline'] * 100:.1f}%")
        t3.metric("Log loss (lager is beter)", f"{metrics['log_loss']:.3f}")
        st.caption("Rond 50% is normaal voor voetbal. Als het model niet duidelijk beter is dan de basislijn, voegt het weinig toe.")
    else:
        st.write("Er is niet genoeg data om het model te testen.")

with st.expander("Over het model en de gegevens"):
    st.write("""
    - Bron: Football-Data.co.uk, uitslagen van de Bundesliga sinds 1993/94.
    - Kenmerken per team: vorm over de laatste 5 en 20 wedstrijden, en vorm thuis of uit.
    - Twee modellen (Random Forest) schatten het verwachte aantal goals van thuis- en uitploeg.
    - Uit die verwachtingen wordt met een Poisson-verdeling een scorematrix berekend.
    - Score, winstkansen, beide teams scoren en meer dan 2,5 goals komen allemaal uit die ene matrix.
    - Blessures, opstellingen en odds worden niet gebruikt.
    - Dit is een prototype en geen advies om geld in te zetten.
    """)

st.caption("Bundesliga Predictor AI • gratis prototype • gemaakt met Python en Streamlit")
