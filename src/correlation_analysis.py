"""
Tests whether earnings call sentiment is a leading indicator of stock price
reaction, using:
  1. Pearson & Spearman correlation (sentiment vs. next-day and 3-day return)
  2. Linear regression (does sentiment meaningfully explain price movement?)
  3. Event-study style breakdown: group calls into sentiment terciles
     (negative / neutral / positive) and compare average returns per group --
     this is the classic technique analysts use to show "does an event class
     produce a distinguishable reaction," not just a raw correlation number
  4. Per-company breakdown: whose sentiment is most/least predictive

Outputs:
  - data/processed/correlation_results.csv
  - data/processed/tercile_analysis.csv
  - data/processed/company_correlations.csv
  - reports/figures/*.png
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr, f_oneway
import statsmodels.api as sm

os.makedirs("reports/figures", exist_ok=True)
sns.set_style("whitegrid")

df = pd.read_csv("data/processed/sentiment_scored.csv")
print(f"Loaded {len(df)} scored, price-merged records\n")

# ---------------------------------------------------------------
# 1. Correlation: sentiment vs. price reaction
# ---------------------------------------------------------------
print("=== Correlation: sentiment_compound vs. price reaction ===")
results = []
for target in ["NextDayReturn", "ThreeDayReturn"]:
    r_p, p_p = pearsonr(df["sentiment_compound"], df[target])
    r_s, p_s = spearmanr(df["sentiment_compound"], df[target])
    print(f"\n{target}:")
    print(f"  Pearson  r = {r_p:.3f}, p = {p_p:.4f} {'(significant)' if p_p < 0.05 else '(not significant)'}")
    print(f"  Spearman r = {r_s:.3f}, p = {p_s:.4f} {'(significant)' if p_s < 0.05 else '(not significant)'}")
    results.append({"target": target, "pearson_r": round(r_p, 3), "pearson_p": round(p_p, 4),
                     "spearman_r": round(r_s, 3), "spearman_p": round(p_s, 4)})
pd.DataFrame(results).to_csv("data/processed/correlation_results.csv", index=False)

# ---------------------------------------------------------------
# 2. Linear regression: how much variance does sentiment explain?
# ---------------------------------------------------------------
print("\n=== Linear regression: NextDayReturn ~ sentiment_compound ===")
X = sm.add_constant(df["sentiment_compound"])
y = df["NextDayReturn"]
model = sm.OLS(y, X).fit()
print(model.summary())
print(f"\nR-squared: {model.rsquared:.3f} -- sentiment alone explains "
      f"{model.rsquared:.1%} of next-day return variance.")

# Save the fitted regression so the app can reuse it for live predictions
import json
with open("models/sentiment_price_model.json", "w", encoding="utf-8") as f:
    json.dump({
        "intercept": float(model.params["const"]),
        "slope": float(model.params["sentiment_compound"]),
        "r_squared": float(model.rsquared),
        "p_value": float(model.pvalues["sentiment_compound"]),
    }, f, indent=2)
print("Saved models/sentiment_price_model.json")

# ---------------------------------------------------------------
# 3. Event-study: sentiment terciles vs. average return
# ---------------------------------------------------------------
print("\n=== Event study: sentiment terciles vs. average next-day return ===")
df["sentiment_tercile"] = pd.qcut(df["sentiment_compound"], 3,
                                    labels=["Negative/cautious call", "Neutral call", "Positive call"])
tercile_summary = df.groupby("sentiment_tercile", observed=True).agg(
    avg_next_day_return=("NextDayReturn", "mean"),
    avg_three_day_return=("ThreeDayReturn", "mean"),
    n_calls=("Ticker", "count")
).round(4)
print(tercile_summary)
tercile_summary.to_csv("data/processed/tercile_analysis.csv")

# ANOVA: is the difference between tercile groups statistically real?
groups = [g["NextDayReturn"].values for _, g in df.groupby("sentiment_tercile", observed=True)]
f_stat, anova_p = f_oneway(*groups)
print(f"\nANOVA across terciles: F = {f_stat:.2f}, p = {anova_p:.4f} "
      f"{'(the groups differ significantly)' if anova_p < 0.05 else '(no significant difference between groups)'}")

# ---------------------------------------------------------------
# 4. Per-company: whose sentiment is most predictive?
# ---------------------------------------------------------------
print("\n=== Per-company correlation: sentiment vs. NextDayReturn ===")
company_results = []
for ticker, g in df.groupby("Ticker"):
    if len(g) >= 5:  # need enough points for a meaningful correlation
        r, p = pearsonr(g["sentiment_compound"], g["NextDayReturn"])
        company_results.append({"Ticker": ticker, "n_quarters": len(g),
                                  "pearson_r": round(r, 3), "p_value": round(p, 4)})
company_df = pd.DataFrame(company_results).sort_values("pearson_r", ascending=False)
print(company_df.to_string(index=False))
company_df.to_csv("data/processed/company_correlations.csv", index=False)

# ---------------------------------------------------------------
# 5. Charts
# ---------------------------------------------------------------
print("\n=== Saving charts to reports/figures/ ===")

fig, ax = plt.subplots(figsize=(7, 5))
sns.regplot(data=df, x="sentiment_compound", y="NextDayReturn", ax=ax,
            scatter_kws={"alpha": 0.6, "color": "#0E6E62"}, line_kws={"color": "#B94A2C"})
ax.set_title("Earnings Call Sentiment vs. Next-Day Stock Return")
ax.set_xlabel("Sentiment Score (VADER compound)")
ax.set_ylabel("Next-Day Return")
plt.tight_layout()
plt.savefig("reports/figures/sentiment_vs_return_scatter.png", dpi=150)
plt.close()

fig, ax = plt.subplots(figsize=(7, 5))
tercile_summary["avg_next_day_return"].plot(kind="bar", ax=ax, color=["#B94A2C", "#C08A2E", "#4C7A5E"])
ax.set_title("Average Next-Day Return by Sentiment Tercile")
ax.set_ylabel("Average Next-Day Return")
ax.axhline(0, color="black", linewidth=0.8)
plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig("reports/figures/tercile_returns.png", dpi=150)
plt.close()

fig, ax = plt.subplots(figsize=(8, 5))
df.groupby("Ticker")["sentiment_compound"].mean().sort_values().plot(kind="barh", ax=ax, color="#8B5E34")
ax.set_title("Average Earnings Call Sentiment by Company")
ax.set_xlabel("Average Sentiment Score")
plt.tight_layout()
plt.savefig("reports/figures/avg_sentiment_by_company.png", dpi=150)
plt.close()

print("Saved 3 chart images.")
print("\n=== DONE ===")
