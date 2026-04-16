import pandas as pd

def get_team_wins(season_year):
    url = f"https://www.basketball-reference.com/leagues/NBA_{season_year}_standings.html"
    tables = pd.read_html(url)
    frames = []
    for t in tables:
        if "W" in t.columns and "L" in t.columns:
            frames.append(t[["Eastern Conference", "W"]].rename(
                columns={"Eastern Conference": "Team"}
            ) if "Eastern Conference" in t.columns else 
            t[["Western Conference", "W"]].rename(
                columns={"Western Conference": "Team"}
            ))
    return pd.concat(frames).assign(Season=season_year)

all_standings = pd.concat([get_team_wins(y) for y in range(2016, 2026)])

# Remove asterisks and strip whitespace
all_standings["Team"] = all_standings["Team"].str.replace(r"\*", "", regex=True).str.strip()

# Remove division header rows
division_names = [
    "Atlantic Division", "Central Division", "Southeast Division",
    "Northwest Division", "Pacific Division", "Southwest Division"
]
all_standings = all_standings[~all_standings["Team"].isin(division_names)]

# Remove any remaining non-team rows (rows where W is not a number)
all_standings = all_standings[pd.to_numeric(all_standings["W"], errors="coerce").notna()]
all_standings["W"] = all_standings["W"].astype(int)

all_standings.to_csv('/Users/aademisai/Desktop/nba_standings.csv', index=False)
print("Saved standings to Desktop!")
print(all_standings.head(10))