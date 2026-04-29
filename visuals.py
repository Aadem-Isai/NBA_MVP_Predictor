

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

hist = pd.read_csv("nba_historical_stats.csv")
curr = pd.read_csv("nba_2026_stats.csv")

# Points per game distribution
plt.figure()
sns.histplot(hist["PTS"])

# Average stats by position
plt.figure()

sns.barplot(data=hist, x="Pos", y="PTS")

# Box plot: scoring spread by position
plt.figure()
sns.boxplot(data=hist, x="Pos", y="PTS")

# Violin plot: shows full distribution shape
plt.figure()
sns.violinplot(data=hist, x="Pos", y="PTS")

plt.figure()
key_stats = ["PTS", "AST", "WS", "W", "BPM", "USG%", "TS%", "PER", "MVP_Score"]
sns.heatmap(hist[key_stats].corr(), annot=True, cmap="coolwarm", fmt=".2f")

plt.figure()
# How scoring peaks and declines with age
sns.lineplot(data=hist, x="Age", y="PTS")

# Multi-stat age curves
for stat in ["PTS", "AST", "TRB"]:
    sns.lineplot(data=hist, x="Age", y=stat, label=stat)

# MVP Score vs PER
plt.figure()
sns.scatterplot(data=hist, x="PER", y="MVP_Score", hue="is_mvp")

plt.figure()

# 2026 efficiency vs volume
sns.scatterplot(data=curr, x="PTS", y="W", hue="Pos", sizes=(50,300))


plt.figure()
sns.histplot(curr["PTS"])

# Average stats by position
plt.figure()

sns.barplot(data=curr, x="Pos", y="PTS")

# Box plot: scoring spread by position
plt.figure()
sns.boxplot(data=curr, x="Pos", y="PTS")

# Violin plot: shows full distribution shape
plt.figure()
sns.violinplot(data=curr, x="Pos", y="PTS")

plt.figure()
key_stats = ["PTS", "AST", "WS", "W", "BPM", "USG%", "TS%", "PER", "MVP_Score"]
sns.heatmap(curr[key_stats].corr(), annot=True, cmap="coolwarm", fmt=".2f")

plt.figure()
# How scoring peaks and declines with age
sns.lineplot(data=curr, x="Age", y="PTS")

# Multi-stat age curves
for stat in ["PTS", "AST", "TRB"]:
    sns.lineplot(data=curr, x="Age", y=stat, label=stat)

# MVP Score vs PER
plt.figure()
sns.scatterplot(data=curr, x="PER", y="MVP_Score")


plt.figure()
# Who's scoring the most this season
top26 = curr.nlargest(15, "PTS")
sns.barplot(data=top26, y="Player", x="PTS")

plt.legend()
plt.show()