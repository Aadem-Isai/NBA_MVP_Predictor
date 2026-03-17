import pandas as pd
import requests
from io import StringIO
from bs4 import BeautifulSoup
import re


def fetch_bbref_table(url, table_id):
    html = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}).text
    html = re.sub(r"<!--(.*?)-->", r"\1", html, flags=re.DOTALL)

    dfs = pd.read_html(StringIO(html), attrs={"id": table_id}, encoding="utf-8")
    df = dfs[0]

    df = df[df["Player"] != "Player"].reset_index(drop=True)
    df = df.drop(columns=["Rk"], errors="ignore")
    df = df.dropna(subset=["Player"])
    df = df.drop(columns=[c for c in df.columns if "Unnamed" in str(c)], errors="ignore")

    # For traded players, keep only the TOT (total) row
    df = df[~((df["Team"] != "TOT") & (df.duplicated(subset=["Player"], keep=False)))]

    numeric_cols = [c for c in df.columns if c not in ["Player", "Pos", "Team", "Age"]]
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
    return df


per_game = fetch_bbref_table(
    "https://www.basketball-reference.com/leagues/NBA_2026_per_game.html",
    "per_game_stats"
)

advanced = fetch_bbref_table(
    "https://www.basketball-reference.com/leagues/NBA_2026_advanced.html",
    "advanced"
)

merged = per_game.merge(advanced, on=["Player", "Team"], suffixes=("", "_adv"))
merged = merged.drop(columns = ["Awards_adv"])

print(merged.shape)
print(merged.head())