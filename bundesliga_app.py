import math
import pandas as pd
import streamlit as st

# Alleen Bundesliga (D1). 2627 is het lopende seizoen; als het bestand nog niet bestaat, wordt het overgeslagen.
SEIZOENEN = ["2223", "2324", "2425", "2526", "2627"]


@st.cache_data(ttl=6 * 3600)
def laad_data():
    frames = []
    for seizoen in SEIZOENEN:
        url = f"https://www.football-data.co.uk/mmz4281/{seizoen}/D1.csv"
        try:
            df = pd.read_csv(url, encoding="latin1")
        except Exception:
            continue
        df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
        df = df[["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"]].dropna()
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True).sort_values("Date")


def bouw_model(data):
    gem_thuis = data["FTHG"].mean()
    gem_uit = data["FTAG"].mean()
    gem = (gem_thuis + gem_uit) / 2
    stats = {}
    for team in sorted(set(data["HomeTeam"]) | set(data["AwayTeam"])):
        thuis = data[data["HomeTeam"] == team]
        uit = data[data["AwayTeam"] == team]
        n = len(thuis) + len(uit)
        voor = thuis["FTHG"].sum() + uit["FTAG"].sum()
        tegen = thuis["FTAG"].sum() + uit["FTHG"].sum()
        stats[team] = {
            "aanval": (voor / n) / gem,
            "verdediging": (tegen / n) / gem,
        }
    return gem_thuis, gem_uit, stats


def poisson_kans(lam, k):
    return math.exp(-lam) * lam ** k / math.factorial(k)


def voorspel(thuis, uit, gem_thuis, gem_uit, stats, max_goals=10):
    lam_h = stats[thuis]["aanval"] * stats[uit]["verdediging"] * gem_thuis
    lam_a = stats[uit]["aanval"] * stats[thuis]["verdediging"] * gem_uit
    p_thuis = p_gelijk = p_uit = 0.0
    beste = (0, 0, 0.0)
    for i in range(max_goals + 1):
        for j in range(max_goals + 1):
            p = poisson_kans(lam_h, i) * poisson_kans(lam_a, j)
            if i > j:
                p_thuis += p
            elif i == j:
                p_gelijk += p
            else:
                p_uit += p
            if p > beste[2]:
                beste = (i, j, p)
    return lam_h, lam_a, p_thuis, p_gelijk, p_uit, beste


st.set_page_config(page_title="Bundesliga-voorspeller", page_icon="⚽")
st.title("⚽ Bundesliga-voorspeller")
st.write(
    "Voorspelt Bundesliga-wedstrijden op basis van historische uitslagen "
    "van football-data.co.uk. Alleen de Bundesliga, geen andere competities."
)

data = laad_data()
if data.empty:
    st.error("Kon de Bundesliga-data niet laden. Controleer de internetverbinding.")
    st.stop()

gem_thuis, gem_uit, stats = bouw_model(data)
teams = sorted(stats.keys())

col1, col2 = st.columns(2)
thuis = col1.selectbox("Thuisploeg", teams, index=0)
uit = col2.selectbox("Uitploeg", teams, index=1)

if thuis == uit:
    st.warning("Kies twee verschillende clubs.")
elif st.button("Voorspel wedstrijd"):
    lam_h, lam_a, ph, pg, pu, beste = voorspel(thuis, uit, gem_thuis, gem_uit, stats)

    st.subheader(f"{thuis} - {uit}")
    c1, c2, c3 = st.columns(3)
    c1.metric("Thuiswinst", f"{ph:.0%}")
    c2.metric("Gelijkspel", f"{pg:.0%}")
    c3.metric("Uitwinst", f"{pu:.0%}")

    st.write(f"Verwachte doelpunten: {thuis} {lam_h:.2f}, {uit} {lam_a:.2f}.")
    st.write(f"Meest waarschijnlijke uitslag: {beste[0]}-{beste[1]} ({beste[2]:.1%}).")
    st.caption(
        "Dit is een eenvoudig statistisch model, geen zekerheid. "
        "Gebruik het niet als basis voor gokbeslissingen."
    )

    onderling = data[
        ((data["HomeTeam"] == thuis) & (data["AwayTeam"] == uit))
        | ((data["HomeTeam"] == uit) & (data["AwayTeam"] == thuis))
    ].tail(5)
    if not onderling.empty:
        st.write("Laatste onderlinge duels:")
        st.dataframe(
            onderling[["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"]],
            hide_index=True,
        )

