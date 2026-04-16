import pandas as pd
import numpy as np
import requests
from io import StringIO
import re
import time
import warnings
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import unicodedata


print("hello world")
def normalize_name(name):
    return unicodedata.normalize("NFD", str(name)).encode("ascii", "ignore").decode("utf-8").strip()


warnings.filterwarnings("ignore")


# ══════════════════════════════════════════════════════════════════════
#  SECTION 5 — BUILD TARGET: winner-anchored MVP score
# ══════════════════════════════════════════════════════════════════════

SCORE_WEIGHTS = {
    "WS":   4.0,
    "BPM":  2.0,
    "PER":  2.5,
    "AST":  0.5,
    "TRB":  0.3,
    "STL":  1.1,
    "BLK":  1.0,
    "PTS":  3.0,
    "W":    11,
}


def build_mvp_score(df, winners):
    df = df.copy()
    df["MVP_Score"] = 0.0
    df["is_mvp"]    = 0

    for season, grp in df.groupby("Season"):
        idx = grp.index

        winner_row = winners[winners["Season"] == season]
        if winner_row.empty:
            continue
        winner_name = winner_row["MVP_Winner"].iloc[0]
        winner_mask = df.loc[idx, "Player"] == winner_name
        df.loc[idx[winner_mask], "is_mvp"] = 1

        season_score = pd.Series(0.0, index=idx)
        for col, w in SCORE_WEIGHTS.items():
            if col not in df.columns:
                continue
            # Ensure values are numeric, coerce errors to NaN, then fill with 0.0
            vals = pd.to_numeric(df.loc[idx, col], errors='coerce').fillna(0.0)
            std  = vals.std()
            z    = (vals - vals.mean()) / std if std > 0 else pd.Series(0.0, index=idx)
            season_score += w * z

        if "W" in df.columns:
            team_wins = df.loc[idx, "W"].fillna(0.0)
            # Scale: 50 wins = no penalty, below 50 = up to 35% penalty
            win_multiplier = (team_wins / 50.0).clip(upper=1.0) * 0.35 + 0.65
            # Now penalizes up to 35% for weak teams
            season_score = season_score * win_multiplier

        if winner_mask.any():
            winner_score = season_score[idx[winner_mask]].iloc[0]
            lo           = season_score.min()
            span         = winner_score - lo
            if span > 0:
                season_score = (season_score - lo) / span
            else:
                season_score = season_score * 0.0
        else:
            lo, hi = season_score.min(), season_score.max()
            season_score = (season_score - lo) / (hi - lo) if hi > lo else season_score * 0.0

        df.loc[idx, "MVP_Score"] = season_score

    return df


# ══════════════════════════════════════════════════════════════════════
#  SECTION 6 — BUILD DATASETS
# ══════════════════════════════════════════════════════════════════════
TRAIN_SEASONS  = list(range(2016, 2025))
PREDICT_SEASON = 2026

print("Loading historical seasons from CSV...")
historical = pd.read_csv("nba_historical_stats.csv")
print(f"Historical dataset shape: {historical.shape}")

print(f"\nLoading {PREDICT_SEASON} season...")
current = pd.read_csv("nba_2026_stats.csv")
print(f"Loaded {len(current)} players for 2026")

# Load standings
standings = pd.read_csv("nba_standings.csv")
standings["Team"] = standings["Team"].str.replace(r"\*", "", regex=True).str.strip()

# Filter division headers
division_names = [
    "Atlantic Division", "Central Division", "Southeast Division",
    "Northwest Division", "Pacific Division", "Southwest Division"
]
standings = standings[~standings["Team"].isin(division_names)]

# Add abbreviations
FULL_TO_ABBREV = {
    "Atlanta Hawks": "ATL", "Boston Celtics": "BOS", "Brooklyn Nets": "BRK",
    "Charlotte Hornets": "CHA", "Chicago Bulls": "CHI", "Cleveland Cavaliers": "CLE",
    "Dallas Mavericks": "DAL", "Denver Nuggets": "DEN", "Detroit Pistons": "DET",
    "Golden State Warriors": "GSW", "Houston Rockets": "HOU", "Indiana Pacers": "IND",
    "Los Angeles Clippers": "LAC", "Los Angeles Lakers": "LAL", "Memphis Grizzlies": "MEM",
    "Miami Heat": "MIA", "Milwaukee Bucks": "MIL", "Minnesota Timberwolves": "MIN",
    "New Orleans Pelicans": "NOP", "New York Knicks": "NYK", "Oklahoma City Thunder": "OKC",
    "Orlando Magic": "ORL", "Philadelphia 76ers": "PHI", "Phoenix Suns": "PHO",
    "Portland Trail Blazers": "POR", "Sacramento Kings": "SAC", "San Antonio Spurs": "SAS",
    "Toronto Raptors": "TOR", "Utah Jazz": "UTA", "Washington Wizards": "WAS",
}
standings["Team_Abbrev"] = standings["Team"].map(FULL_TO_ABBREV)

# Add 2026 standings BEFORE merging
standings_2026 = pd.DataFrame({
    "Team": [
        "Oklahoma City Thunder", "Cleveland Cavaliers", "Boston Celtics",
        "New York Knicks", "Memphis Grizzlies", "Los Angeles Lakers",
        "Golden State Warriors", "Houston Rockets", "Denver Nuggets",
        "Minnesota Timberwolves", "Los Angeles Clippers", "Indiana Pacers",
        "Dallas Mavericks", "Miami Heat", "Milwaukee Bucks",
        "Sacramento Kings", "Phoenix Suns", "San Antonio Spurs",
        "New Orleans Pelicans", "Utah Jazz", "Toronto Raptors",
        "Atlanta Hawks", "Orlando Magic", "Chicago Bulls",
        "Brooklyn Nets", "Charlotte Hornets", "Philadelphia 76ers",
        "Detroit Pistons", "Portland Trail Blazers", "Washington Wizards"
    ],
    "W": [68, 64, 61, 55, 49, 48, 44, 52, 45, 50,
          40, 50, 43, 42, 38, 40, 35, 34, 30, 22,
          25, 38, 41, 28, 20, 30, 24, 42, 20, 15]
})
standings_2026["Season"] = 2026
standings
standings_2026["Team_Abbrev"] = standings_2026["Team"].map(FULL_TO_ABBREV)
standings = pd.concat([standings, standings_2026], ignore_index=True)

# Verify
unmapped = standings[standings["Team_Abbrev"].isna()]["Team"].unique()
print("Unmapped teams:", unmapped)

# Fix CHO before merging
current["Team"] = current["Team"].replace({"CHO": "CHA"})

# Drop any existing W column to prevent W_x/W_y collision
historical = historical.drop(columns=["W"], errors="ignore")
current    = current.drop(columns=["W"], errors="ignore")

# NOW merge
historical = historical.merge(
    standings[["Season", "Team_Abbrev", "W"]],
    left_on=["Season", "Team"],
    right_on=["Season", "Team_Abbrev"],
    how="left"
).drop(columns="Team_Abbrev")

# Force 'W' to numeric and print non-numeric values
non_numeric_hist = historical[~historical['W'].apply(lambda x: str(x).replace('.', '', 1).isdigit()) & ~historical['W'].isna()]
historical['W'] = pd.to_numeric(historical['W'], errors='coerce')

current = current.merge(
    standings[["Season", "Team_Abbrev", "W"]],
    left_on=["Season", "Team"],
    right_on=["Season", "Team_Abbrev"],
    how="left"
).drop(columns="Team_Abbrev")

# Force 'W' to numeric and print non-numeric values
non_numeric_curr = current[~current['W'].apply(lambda x: str(x).replace('.', '', 1).isdigit()) & ~current['W'].isna()]
current['W'] = pd.to_numeric(current['W'], errors='coerce')

print(current[current["Player"].isin(["Nikola Jokić", "Shai Gilgeous-Alexander"])][["Player", "Team", "W"]])

mvp_winners = pd.DataFrame({
    "Season": [2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025],
    "MVP_Winner": [
        "Stephen Curry", "Russell Westbrook", "James Harden",
        "Giannis Antetokounmpo", "Giannis Antetokounmpo", "Nikola Jokic",
        "Nikola Jokic", "Joel Embiid", "Nikola Jokic", "Shai Gilgeous-Alexander"
    ]
})

historical = build_mvp_score(historical, mvp_winners)

# ══════════════════════════════════════════════════════════════════════
#  SECTION 7 — MODEL
# ══════════════════════════════════════════════════════════════════════

SCORING    = ["PTS", "FG%", "3P%", "FT%", "TS%"]
PLAYMAKING = ["AST", "TOV"]
DEFENSE    = ["TRB", "STL", "BLK"]
EFFICIENCY = ["PER", "WS/48", "BPM", "VORP"]
USAGE      = ["MP", "G", "W"]  
FEATURES   = SCORING + PLAYMAKING + DEFENSE + EFFICIENCY + USAGE
TARGET     = "MVP_Score"


def prep(df, features, has_target=True):
    df = df.copy()
    df = df[(df["G"] >= 64) & (df["MP"] >= 20) & (df["PTS"] >= 15)]
    if "TOV" in df.columns:
        df["TOV"] = df["TOV"].max() - df["TOV"]
    drop_subset = features + [TARGET] if has_target else features
    # Only drop rows missing features that actually exist in the df
    existing = [c for c in drop_subset if c in df.columns]
    return df.dropna(subset=existing)


hist_clean = prep(historical, FEATURES, has_target=True)
curr_clean = prep(current.assign(MVP_Score=0.0), FEATURES, has_target=False)

train_df = hist_clean[hist_clean["Season"] <= 2024]
test_df  = hist_clean[hist_clean["Season"] == 2025]

# Only use features that exist in both train and current data
FEATURES = [f for f in FEATURES if f in train_df.columns and f in curr_clean.columns]

X_train, y_train = train_df[FEATURES], train_df[TARGET]
X_test,  y_test  = test_df[FEATURES],  test_df[TARGET]
X_pred           = curr_clean[FEATURES]

scaler    = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)
X_pred_s  = scaler.transform(X_pred)

model = GradientBoostingRegressor(
    n_estimators=400, learning_rate=0.04,
    max_depth=5, subsample=0.8,
    min_samples_leaf=3, random_state=42
)
model.fit(X_train_s, y_train)

# ── Evaluate on 2025 ─────────────────────────────────────────────────
# Use raw (unclipped) predictions for ranking 
y_pred_test_raw = model.predict(X_test_s)
y_pred_test = y_pred_test_raw

test_df = test_df.copy()
test_df["Predicted_Score"] = y_pred_test
test_df["_raw_pred"]       = y_pred_test_raw

rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
r2   = r2_score(y_test, y_pred_test)

# Rank on raw predictions
test_df["Pred_Rank"]   = test_df["_raw_pred"].rank(ascending=False, method="first").astype(int)
test_df["Actual_Rank"] = test_df[TARGET].rank(ascending=False, method="first").astype(int)

predicted_mvp_2025 = normalize_name(test_df.loc[test_df["_raw_pred"].idxmax(), "Player"])
actual_mvp_2025    = normalize_name(mvp_winners.loc[mvp_winners["Season"] == 2025, "MVP_Winner"].iloc[0])

print("\n" + "=" * 50)
print(f"  2025 TEST RESULTS")
print(f"  RMSE:              {rmse:.4f}")
print(f"  R2:                {r2:.4f}")
print(f"  Predicted #1:      {predicted_mvp_2025}")
print(f"  Actual MVP:        {actual_mvp_2025}")
print(f"  Correct:           {predicted_mvp_2025 == actual_mvp_2025}")
print("=" * 50)

# ── Historical accuracy check ─────────────────────────────────────────
print("\nHistorical accuracy check (train seasons):")
correct = 0
season_list = sorted(hist_clean[hist_clean["Season"] <= 2024]["Season"].unique())
for season in season_list:
    grp = hist_clean[hist_clean["Season"] == season].copy()
    if grp.empty:
        continue
    raw_preds    = model.predict(scaler.transform(grp[FEATURES]))
    pred_winner  = grp.iloc[np.argmax(raw_preds)]["Player"]
    actual_row   = mvp_winners[mvp_winners["Season"] == season]
    if actual_row.empty:
        continue
    actual_winner = actual_row["MVP_Winner"].iloc[0]
    match = normalize_name(pred_winner) == normalize_name(actual_winner)
    correct += int(match)
    print(f"  {season}  Predicted: {pred_winner:<28} Actual: {actual_winner:<28} {'✓' if match else '✗'} Predicted Score: {max(raw_preds):.4f}")

total = len(season_list)
print(f"\n  Train accuracy: {correct}/{total} = {correct/total:.0%}")

# ── Predict 2026 ─────────────────────────────────────────────────────
raw_pred_2026 = model.predict(X_pred_s)
curr_clean = curr_clean.copy()
curr_clean["Predicted_Score"] = raw_pred_2026
# Rank on raw scores
curr_clean["MVP_Rank"] = pd.Series(raw_pred_2026).rank(ascending=False, method="first").astype(int).values
curr_clean = curr_clean.sort_values("MVP_Rank")

print("\nTop 10 Predicted 2026 MVP Candidates:")
print(
    curr_clean[["MVP_Rank", "Player", "Team", "PTS","TRB", "AST", "Predicted_Score"]]
    .head(10)
    .to_string(index=False)
)


# ══════════════════════════════════════════════════════════════════════
#  SECTION 8 — VISUALS
# ══════════════════════════════════════════════════════════════════════

sns.set_theme(style="darkgrid", palette="muted")
GOLD = "#FFD700"
NAVY = "#1D428A"
RED  = "#C8102E"
BLUE = "#4C8BF5"

fig = plt.figure(figsize=(20, 24))
fig.suptitle("NBA MVP Predictor — Gradient Boosting Model",
             fontsize=22, fontweight="bold", y=0.98)

# ── Plot 1: Feature Importance ────────────────────────────────────────
ax1 = fig.add_subplot(3, 2, 1)
imp_df = pd.DataFrame({"Feature": FEATURES, "Importance": model.feature_importances_})
imp_df = imp_df.sort_values("Importance", ascending=True)
top3   = imp_df.tail(3)["Feature"].values
bar_colors = [GOLD if f in top3 else BLUE for f in imp_df["Feature"]]
ax1.barh(imp_df["Feature"], imp_df["Importance"],
         color=bar_colors, edgecolor="white", linewidth=0.5)
ax1.set_title("Feature Importance  (gold = top 3)", fontweight="bold", fontsize=13)
ax1.set_xlabel("Importance Score")
ax1.tick_params(axis="y", labelsize=8)

# ── Plot 2: Predicted vs Actual Score — 2025 ─────────────────────────
ax2 = fig.add_subplot(3, 2, 2)
sc = ax2.scatter(y_test, y_pred_test, alpha=0.7, s=60,
                 c=y_pred_test, cmap="YlOrRd", edgecolors="gray", linewidth=0.4)
plt.colorbar(sc, ax=ax2, label="Predicted Score")
lims = [min(y_test.min(), y_pred_test.min()) - 0.02,
        max(y_test.max(), y_pred_test.max()) + 0.02]
ax2.plot(lims, lims, "--", color="gray", alpha=0.6, linewidth=1.2, label="Perfect")
for _, row in test_df.nlargest(5, TARGET).iterrows():
    ax2.annotate(row["Player"].split(" ")[-1],
                 (row[TARGET], row["Predicted_Score"]),
                 textcoords="offset points", xytext=(6, 4),
                 fontsize=8, color=RED, fontweight="bold")
ax2.set_xlabel("Actual MVP Score")
ax2.set_ylabel("Predicted MVP Score")
ax2.set_title(f"Predicted vs Actual Score — 2025 Test\nRMSE={rmse:.4f}  R2={r2:.4f}",
              fontweight="bold", fontsize=13)
ax2.legend()

# ── Plot 3: Top 10 2026 Candidates ───────────────────────────────────
ax3 = fig.add_subplot(3, 2, 3)
top10 = curr_clean.head(10).copy()
bcolors = [GOLD if i == 0 else BLUE for i in range(len(top10))]
ax3.barh(top10["Player"][::-1], top10["Predicted_Score"][::-1],
         color=bcolors[::-1], edgecolor="white", linewidth=0.5)
ax3.set_title("Top 10 Predicted 2026 MVP Candidates  (gold = #1)",
              fontweight="bold", fontsize=13)
ax3.set_xlabel("Predicted MVP Score (0-1)")
for i, (_, row) in enumerate(top10[::-1].iterrows()):
    ax3.text(0.01, i, f"#{int(row['MVP_Rank'])}",
             va="center", fontsize=8, color="white", fontweight="bold")

# ── Plot 4: Correlation heatmap ───────────────────────────────────────
ax4 = fig.add_subplot(3, 2, 4)
KEY_FEATS = ["PTS", "AST", "TRB", "WS", "BPM", "VORP", "MVP_Score"]
corr_data = historical[KEY_FEATS].dropna().corr()
mask = np.triu(np.ones_like(corr_data, dtype=bool))
sns.heatmap(corr_data, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
            center=0, linewidths=0.5, ax=ax4,
            annot_kws={"size": 8}, cbar_kws={"shrink": 0.8})
ax4.set_title("Feature Correlation w/ MVP Score", fontweight="bold", fontsize=13)
ax4.tick_params(axis="x", rotation=45, labelsize=8)
ax4.tick_params(axis="y", rotation=0,  labelsize=8)

# ── Plot 5: Actual MVP score each season ─────────────────────────────
ax5 = fig.add_subplot(3, 2, 5)
mvp_scores_per_season = []
for _, row in mvp_winners[mvp_winners["Season"].isin(TRAIN_SEASONS)].iterrows():
    match = historical[
        (historical["Season"] == row["Season"]) &
        (historical["Player"] == row["MVP_Winner"])
    ]
    if not match.empty:
        mvp_scores_per_season.append({
            "Season": row["Season"],
            "Player": row["MVP_Winner"],
            "MVP_Score": match["MVP_Score"].iloc[0]
        })
mvp_plot_df = pd.DataFrame(mvp_scores_per_season).sort_values("Season")
bars = ax5.bar(mvp_plot_df["Season"], mvp_plot_df["MVP_Score"],
               color=NAVY, edgecolor="white", linewidth=0.5)
for bar, (_, row) in zip(bars, mvp_plot_df.iterrows()):
    ax5.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
             row["Player"].split(" ")[-1],
             ha="center", va="bottom", fontsize=7.5, fontweight="bold", rotation=35)
ax5.set_title("Actual MVP's Stat Score Each Season", fontweight="bold", fontsize=13)
ax5.set_xlabel("Season End Year")
ax5.set_ylabel("MVP Score (winner anchored at 1.0)")
ax5.set_ylim(0, 1.15)
ax5.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig("mvp_model_results.png", dpi=150, bbox_inches="tight")
plt.show()
print("\nSaved -> mvp_model_results.png")
