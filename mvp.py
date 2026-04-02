import pandas as pd
import requests
from io import StringIO
from bs4 import BeautifulSoup
import re


def fetch_bbref_table(url, table_id):
    r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
    r.encoding = "utf-8"
    html = r.text
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

def fetch_bbref_standings(url):
    r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
    r.encoding = "utf-8"
    html = r.text
    html = re.sub(r"<!--(.*?)-->", r"\1", html, flags=re.DOTALL)

    east = pd.read_html(StringIO(html), attrs={"id": "confs_standings_E"}, encoding="utf-8")[0]
    west = pd.read_html(StringIO(html), attrs={"id": "confs_standings_W"}, encoding="utf-8")[0]

    # Rename each table's team column to "Team_full" before concat
    east = east.rename(columns={east.columns[0]: "Team_full"})[["Team_full", "W", "L"]]
    west = west.rename(columns={west.columns[0]: "Team_full"})[["Team_full", "W", "L"]]

    standings = pd.concat([east, west], ignore_index=True)
    standings = standings[pd.to_numeric(standings["W"], errors="coerce").notna()]
    standings["total_games"] = standings["W"].astype(int) + standings["L"].astype(int)

    standings["Team_full"] = standings["Team_full"].str.replace(r"\s*\(\d+\)", "", regex=True).str.replace(r"\*", "", regex=True).str.strip()

    name_to_abbrev = {
        "Atlanta Hawks": "ATL", "Boston Celtics": "BOS", "Brooklyn Nets": "BRK",
        "Charlotte Hornets": "CHO", "Chicago Bulls": "CHI", "Cleveland Cavaliers": "CLE",
        "Dallas Mavericks": "DAL", "Denver Nuggets": "DEN", "Detroit Pistons": "DET",
        "Golden State Warriors": "GSW", "Houston Rockets": "HOU", "Indiana Pacers": "IND",
        "Los Angeles Clippers": "LAC", "Los Angeles Lakers": "LAL", "Memphis Grizzlies": "MEM",
        "Miami Heat": "MIA", "Milwaukee Bucks": "MIL", "Minnesota Timberwolves": "MIN",
        "New Orleans Pelicans": "NOP", "New York Knicks": "NYK", "Oklahoma City Thunder": "OKC",
        "Orlando Magic": "ORL", "Philadelphia 76ers": "PHI", "Phoenix Suns": "PHO",
        "Portland Trail Blazers": "POR", "Sacramento Kings": "SAC", "San Antonio Spurs": "SAS",
        "Toronto Raptors": "TOR", "Utah Jazz": "UTA", "Washington Wizards": "WAS",
    }

    standings["Team"] = standings["Team_full"].map(name_to_abbrev)
    return standings[["Team", "total_games"]]


per_game = fetch_bbref_table(
    "https://www.basketball-reference.com/leagues/NBA_2026_per_game.html",
    "per_game_stats"
)

advanced = fetch_bbref_table(
    "https://www.basketball-reference.com/leagues/NBA_2026_advanced.html",
    "advanced"
)

merged = per_game.merge(advanced, on=["Player", "Team"], suffixes=("", "_adv"))
merged = merged.drop(columns=["Awards_adv"])
merged = merged.drop(columns = ["Awards"])

standings = fetch_bbref_standings("https://www.basketball-reference.com/leagues/NBA_2026.html")
merged = merged.merge(standings, on="Team", how="left")
merged = merged[merged['G'] >= 0.8 * merged['total_games']]

print(merged.shape)
print(merged)
merged.to_csv("nba_2026_stats.csv", index=False)