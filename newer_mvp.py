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

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════════
#  SECTION 1 — SCRAPERS
# ══════════════════════════════════════════════════════════════════════

def _flatten_columns(df):
    if not isinstance(df.columns, pd.MultiIndex):
        return df
    bottom = [col[-1].strip() for col in df.columns]
    if len(bottom) == len(set(bottom)):
        df.columns = bottom
        return df
    new_cols = []
    for col in df.columns:
        top, bot = col[0].strip(), col[-1].strip()
        new_cols.append(bot if top.lower().startswith("unnamed") else f"{top} {bot}")
    df.columns = new_cols
    return df


def fetch_bbref_table(url, table_id):
    r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
    r.encoding = "utf-8"
    html = re.sub(r"<!--(.*?)-->", r"\1", r.text, flags=re.DOTALL)

    # Try the exact ID first, then common suffix variants
    candidate_ids = [table_id, table_id.replace("_stats", ""), f"div_{table_id}"]
    df = None
    for tid in candidate_ids:
        try:
            df = pd.read_html(StringIO(html), attrs={"id": tid}, encoding="utf-8")[0]
            break
        except ValueError:
            continue

    # Fallback: scan all tables for a Player column (stats pages always have one)
    if df is None:
        print(f"  WARNING: table id='{table_id}' not found at {url} — scanning all tables...")
        all_tables = pd.read_html(StringIO(html), encoding="utf-8")
        for t in all_tables:
            t = _flatten_columns(t)
            if "Player" in t.columns and len(t) > 20:
                df = t
                break

    if df is None:
        raise RuntimeError(f"No usable table found at {url}")

    df = _flatten_columns(df)
    df = df.drop(columns=["Rk", "Rank"], errors="ignore")
    df = df.drop(columns=[c for c in df.columns if "Unnamed" in str(c)], errors="ignore")
    df = df.rename(columns={"Tm": "Team"}, errors="ignore")

    if "Player" in df.columns:
        df = df[df["Player"] != "Player"].reset_index(drop=True)
        df = df.dropna(subset=["Player"])

    if "Team" in df.columns:
        df = df[~((df["Team"] != "TOT") & (df.duplicated(subset=["Player"], keep=False)))]

    numeric_cols = [c for c in df.columns if c not in ["Player", "Pos", "Team", "Age"]]
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
    return df.reset_index(drop=True)


def fetch_bbref_standings(url):
    r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
    r.encoding = "utf-8"
    html = re.sub(r"<!--(.*?)-->", r"\1", r.text, flags=re.DOTALL)

    east = _flatten_columns(pd.read_html(StringIO(html), attrs={"id": "confs_standings_E"}, encoding="utf-8")[0])
    west = _flatten_columns(pd.read_html(StringIO(html), attrs={"id": "confs_standings_W"}, encoding="utf-8")[0])

    def _clean(df):
        df = df.rename(columns={df.columns[0]: "Team_full"})[["Team_full", "W", "L"]]
        df = df[pd.to_numeric(df["W"], errors="coerce").notna()].copy()
        df["W"] = df["W"].astype(int)
        df["L"] = df["L"].astype(int)
        df["total_games"] = df["W"] + df["L"]
        df["Team_full"] = (
            df["Team_full"]
            .str.replace(r"\s*\(\d+\)", "", regex=True)
            .str.replace(r"\*", "", regex=True)
            .str.strip()
        )
        return df

    standings = pd.concat([_clean(east), _clean(west)], ignore_index=True)

    name_to_abbrev = {
        "Atlanta Hawks": "ATL",          "Boston Celtics": "BOS",
        "Brooklyn Nets": "BRK",          "Charlotte Hornets": "CHO",
        "Chicago Bulls": "CHI",          "Cleveland Cavaliers": "CLE",
        "Dallas Mavericks": "DAL",       "Denver Nuggets": "DEN",
        "Detroit Pistons": "DET",        "Golden State Warriors": "GSW",
        "Houston Rockets": "HOU",        "Indiana Pacers": "IND",
        "Los Angeles Clippers": "LAC",   "Los Angeles Lakers": "LAL",
        "Memphis Grizzlies": "MEM",      "Miami Heat": "MIA",
        "Milwaukee Bucks": "MIL",        "Minnesota Timberwolves": "MIN",
        "New Orleans Pelicans": "NOP",   "New York Knicks": "NYK",
        "Oklahoma City Thunder": "OKC",  "Orlando Magic": "ORL",
        "Philadelphia 76ers": "PHI",     "Phoenix Suns": "PHO",
        "Portland Trail Blazers": "POR", "Sacramento Kings": "SAC",
        "San Antonio Spurs": "SAS",      "Toronto Raptors": "TOR",
        "Utah Jazz": "UTA",              "Washington Wizards": "WAS",
    }
    standings["Team"] = standings["Team_full"].map(name_to_abbrev)
    return standings[["Team", "W", "L", "total_games"]]


# ══════════════════════════════════════════════════════════════════════
#  SECTION 2 — MVP WINNER SCRAPER  (names only, zero vote data)
# ══════════════════════════════════════════════════════════════════════

def fetch_mvp_winners():
    print("Fetching MVP winner list (names only)...")
    url = "https://www.basketball-reference.com/awards/mvp.html"
    r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
    r.encoding = "utf-8"
    html = re.sub(r"<!--(.*?)-->", r"\1", r.text, flags=re.DOTALL)

    # Try known table IDs — BBRef has used several over the years
    candidate_ids = ["mvp_NBA", "mvp", "awards_mvp", "nba_mvp"]
    df = None
    for tid in candidate_ids:
        try:
            df = pd.read_html(StringIO(html), attrs={"id": tid}, encoding="utf-8")[0]
            print(f"  Found MVP table using id='{tid}'")
            break
        except ValueError:
            continue

    # Last resort: grab ALL tables and find the one with Season + Player columns
    if df is None:
        print("  Named table not found — scanning all tables for Season+Player columns...")
        all_tables = pd.read_html(StringIO(html), encoding="utf-8")
        for t in all_tables:
            t = _flatten_columns(t)
            if "Season" in t.columns and "Player" in t.columns:
                df = t
                print("  Found MVP table by column content")
                break

    if df is None:
        raise RuntimeError(
            "Could not find the MVP table on basketball-reference.com. "
            "The page structure may have changed significantly — check the URL manually."
        )

    df = _flatten_columns(df)

    # Filter out header rows that repeat inside the table
    if "Season" in df.columns:
        df = df[df["Season"] != "Season"].reset_index(drop=True)
    df = df.dropna(subset=["Season", "Player"])

    # Season column looks like "2024-25" → convert to end year (2025)
    df["Season"] = df["Season"].str.extract(r"(\d{4})-\d+")[0].astype(int) + 1

    winners = (
        df.drop_duplicates(subset="Season", keep="first")[["Season", "Player"]]
        .rename(columns={"Player": "MVP_Winner"})
        .copy()
    )
    print(f"  Got {len(winners)} MVP winners\n")
    return winners


# ══════════════════════════════════════════════════════════════════════
#  SECTION 3 — PER-SEASON FETCHER
# ══════════════════════════════════════════════════════════════════════

def fetch_season(year):
    print(f"  Fetching {year}...")
    base = f"https://www.basketball-reference.com/leagues/NBA_{year}"

    per_game  = fetch_bbref_table(f"{base}_per_game.html", "per_game_stats")
    time.sleep(4)
    advanced  = fetch_bbref_table(f"{base}_advanced.html", "advanced")
    time.sleep(4)
    standings = fetch_bbref_standings(f"{base}.html")
    time.sleep(4)

    merged = per_game.merge(advanced, on=["Player", "Team"], suffixes=("", "_adv"))
    merged = merged.drop(columns=["Awards_adv", "Awards"], errors="ignore")
    merged = merged.merge(standings, on="Team", how="left")

    merged["win_pct"]  = merged["W"] / merged["total_games"]
    # Games played as a fraction of team games (used as a feature, not a hard filter)
    merged["games_pct"] = merged["G"] / merged["total_games"]

    # Eligibility: lowered to 65% games (catches Embiid-type seasons) + 20 MPG
    merged = merged[
        (merged["G"]  >= 0.65 * merged["total_games"]) &
        (merged["MP"] >= 20.0)
    ].reset_index(drop=True)

    merged["Season"] = year
    print(f"    -> {len(merged)} eligible players")
    return merged


# ══════════════════════════════════════════════════════════════════════
#  SECTION 4 — FEATURE ENGINEERING: within-season percentile ranks
# ══════════════════════════════════════════════════════════════════════
#
#  For each key stat we add a "_pct" column = percentile rank within
#  that season (0–1, higher = better). This tells the model not just
#  "this player scored 30 PPG" but "this player was in the 99th
#  percentile for scoring that season" — capturing relative dominance,
#  which is what voters actually respond to.

PERCENTILE_COLS = ["PTS", "BPM", "VORP", "WS", "win_pct", "TS%", "PER"]

def add_percentile_ranks(df):
    df = df.copy()
    for season, grp in df.groupby("Season"):
        idx = grp.index
        for col in PERCENTILE_COLS:
            if col not in df.columns:
                continue
            vals = df.loc[idx, col]
            pct_col = f"{col}_pct"
            # percentile rank within season (0–1)
            df.loc[idx, pct_col] = vals.rank(pct=True)
    return df


# ══════════════════════════════════════════════════════════════════════
#  SECTION 5 — BUILD TARGET: winner-anchored MVP score
# ══════════════════════════════════════════════════════════════════════

SCORE_WEIGHTS = {
    "VORP":    3.0,
    "WS":      2.5,
    "BPM":     2.0,
    "win_pct": 2.5,
    "PER":     1.5,
    "PTS":     1.2,
    "TS%":     1.3,
    "AST":     1.0,
    "TRB":     0.8,
    "STL":     0.7,
    "BLK":     0.7,
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
            vals = df.loc[idx, col].fillna(0.0)
            std  = vals.std()
            z    = (vals - vals.mean()) / std if std > 0 else pd.Series(0.0, index=idx)
            season_score += w * z

        # Anchor: winner = 1.0, scale everyone else relative to that
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

        df.loc[idx, "MVP_Score"] = season_score.clip(lower=0.0)

    return df


# ══════════════════════════════════════════════════════════════════════
#  SECTION 6 — BUILD DATASETS
# ══════════════════════════════════════════════════════════════════════

TRAIN_SEASONS  = list(range(2016, 2026))
PREDICT_SEASON = 2026

mvp_winners = fetch_mvp_winners()

print("Fetching historical seasons...")
all_seasons = []
for year in TRAIN_SEASONS:
    try:
        all_seasons.append(fetch_season(year))
    except Exception as e:
        print(f"  ERROR on {year}: {e}")

historical = pd.concat(all_seasons, ignore_index=True)
historical = add_percentile_ranks(historical)
historical = build_mvp_score(historical, mvp_winners)

print(f"\nHistorical dataset shape: {historical.shape}")
historical.to_csv("nba_historical_stats.csv", index=False)

print(f"\nFetching {PREDICT_SEASON} season...")
current = fetch_season(PREDICT_SEASON)
current = add_percentile_ranks(current)
current["MVP_Score"] = None
current["is_mvp"]    = None
current.to_csv("nba_2026_stats.csv", index=False)
print("Saved nba_2026_stats.csv")


# ══════════════════════════════════════════════════════════════════════
#  SECTION 7 — MODEL
# ══════════════════════════════════════════════════════════════════════

PCT_FEATURES = [f"{c}_pct" for c in PERCENTILE_COLS]

SCORING    = ["PTS", "FG%", "3P%", "FT%", "TS%"]
PLAYMAKING = ["AST", "TOV"]
DEFENSE    = ["TRB", "STL", "BLK"]
EFFICIENCY = ["PER", "WS", "WS/48", "BPM", "VORP"]
TEAM       = ["win_pct"]
USAGE      = ["MP", "G", "games_pct"]   # games_pct lets model learn availability signal
FEATURES   = SCORING + PLAYMAKING + DEFENSE + EFFICIENCY + TEAM + USAGE + PCT_FEATURES
TARGET     = "MVP_Score"


def prep(df, features, has_target=True):
    df = df.copy()
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
    max_depth=4, subsample=0.8,
    min_samples_leaf=3, random_state=42
)
model.fit(X_train_s, y_train)

# ── Evaluate on 2025 ─────────────────────────────────────────────────
# Use raw (unclipped) predictions for ranking — clipping causes ties at 1.0
y_pred_test_raw = model.predict(X_test_s)
y_pred_test     = np.clip(y_pred_test_raw, 0, 1)

test_df = test_df.copy()
test_df["Predicted_Score"] = y_pred_test
test_df["_raw_pred"]       = y_pred_test_raw   # used for tie-free ranking

rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
r2   = r2_score(y_test, y_pred_test)

# Rank on raw predictions — guarantees no ties
test_df["Pred_Rank"]   = test_df["_raw_pred"].rank(ascending=False, method="first").astype(int)
test_df["Actual_Rank"] = test_df[TARGET].rank(ascending=False, method="first").astype(int)

predicted_mvp_2025 = test_df.loc[test_df["_raw_pred"].idxmax(), "Player"]
actual_mvp_2025    = mvp_winners.loc[mvp_winners["Season"] == 2025, "MVP_Winner"].iloc[0]

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
    match = pred_winner == actual_winner
    correct += int(match)
    print(f"  {season}  Predicted: {pred_winner:<28} Actual: {actual_winner:<28} {'✓' if match else '✗'}")

total = len(season_list)
print(f"\n  Train accuracy: {correct}/{total} = {correct/total:.0%}")

# ── Predict 2026 ─────────────────────────────────────────────────────
raw_pred_2026 = model.predict(X_pred_s)
curr_clean = curr_clean.copy()
curr_clean["Predicted_Score"] = np.clip(raw_pred_2026, 0, 1)
# Rank on raw scores — no ties possible
curr_clean["MVP_Rank"] = pd.Series(raw_pred_2026).rank(ascending=False, method="first").astype(int).values
curr_clean = curr_clean.sort_values("MVP_Rank")

print("\nTop 10 Predicted 2026 MVP Candidates:")
print(
    curr_clean[["MVP_Rank", "Player", "Team", "PTS", "win_pct", "Predicted_Score"]]
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
KEY_FEATS = ["PTS", "AST", "TRB", "WS", "BPM", "VORP", "win_pct", "MVP_Score"]
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

# ── Plot 6: Predicted vs Score Rank — 2025 ───────────────────────────
ax6 = fig.add_subplot(3, 2, 6)
top10_ranks = test_df.nsmallest(10, "Pred_Rank")[["Player", "Pred_Rank", "Actual_Rank"]].sort_values("Pred_Rank")
x = np.arange(len(top10_ranks))
w = 0.35
ax6.bar(x - w/2, top10_ranks["Pred_Rank"],   w, label="Predicted Rank", color=BLUE, alpha=0.85)
ax6.bar(x + w/2, top10_ranks["Actual_Rank"], w, label="Score Rank",     color=RED,  alpha=0.85)
ax6.set_xticks(x)
ax6.set_xticklabels([p.split(" ")[-1] for p in top10_ranks["Player"]],
                    rotation=35, ha="right", fontsize=9)
ax6.invert_yaxis()
ax6.set_ylabel("Rank  (1 = best)")
ax6.set_title("Predicted vs Score Rank — 2025 Season", fontweight="bold", fontsize=13)
ax6.legend()

plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig("mvp_model_results.png", dpi=150, bbox_inches="tight")
plt.show()
print("\nSaved -> mvp_model_results.png")
