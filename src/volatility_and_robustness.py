"""
Two extra analyses:

1. Earnings-day volatility by company
   Which stocks react most violently to earnings calls, regardless of tone?
   (standard deviation of next-day return, plus average absolute move)

2. Bootstrap robustness check
   With only 64 data points, a single correlation number can be fragile.
   We resample the data 2,000 times and see how much the correlation between
   sentiment and next-day return wobbles. If the 95% interval comfortably
   includes zero, that is honest evidence the market-wide signal is weak.

Output:
  - data/processed/volatility_by_company.csv
  - data/processed/bootstrap_results.csv
  - reports/figures/volatility_by_company.png
  - reports/figures/bootstrap_correlation.png
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

os.makedirs("reports/figures", exist_ok=True)
np.random.seed(42)

df = pd.read_csv("data/processed/sentiment_scored.csv")

# ---------------------------------------------------------------
# 1. Volatility by company
# ---------------------------------------------------------------
print("=== Earnings-day reaction volatility by company ===")
vol = df.groupby("Ticker").agg(
    std_next_day=("NextDayReturn", "std"),
    avg_abs_move=("NextDayReturn", lambda s: s.abs().mean()),
    worst_day=("NextDayReturn", "min"),
    best_day=("NextDayReturn", "max"),
).round(4).sort_values("std_next_day", ascending=False)
print(vol)
vol.to_csv("data/processed/volatility_by_company.csv")

fig, ax = plt.subplots(figsize=(8, 5))
vol["avg_abs_move"].sort_values().plot(kind="barh", ax=ax, color="#217346")
ax.set_title("Average Absolute Next-Day Move After Earnings")
ax.set_xlabel("Average |return|")
plt.tight_layout()
plt.savefig("reports/figures/volatility_by_company.png", dpi=150)
plt.close()

# ---------------------------------------------------------------
# 2. Bootstrap the sentiment/return correlation
# ---------------------------------------------------------------
print("\n=== Bootstrap: how stable is the sentiment-return correlation? ===")
n = len(df)
boot_corrs = []
for _ in range(2000):
    sample = df.sample(n=n, replace=True)
    r, _ = pearsonr(sample["sentiment_compound"], sample["NextDayReturn"])
    boot_corrs.append(r)
boot_corrs = np.array(boot_corrs)

observed_r, _ = pearsonr(df["sentiment_compound"], df["NextDayReturn"])
low, high = np.percentile(boot_corrs, [2.5, 97.5])
share_positive = (boot_corrs > 0).mean()

print(f"Observed correlation:      {observed_r:.3f}")
print(f"95% bootstrap interval:    [{low:.3f}, {high:.3f}]")
print(f"Share of resamples > 0:    {share_positive:.1%}")
if low <= 0 <= high:
    print("-> The interval includes zero: the market-wide relationship is NOT "
          "reliably distinguishable from no relationship at this sample size.")
else:
    print("-> The interval excludes zero: the relationship holds up under resampling.")

pd.DataFrame([{
    "observed_r": round(observed_r, 3), "ci_low": round(low, 3),
    "ci_high": round(high, 3), "share_resamples_positive": round(share_positive, 3),
}]).to_csv("data/processed/bootstrap_results.csv", index=False)

fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(boot_corrs, bins=40, color="#A9A9A9", edgecolor="white")
ax.axvline(observed_r, color="#217346", linewidth=2, label=f"Observed r = {observed_r:.2f}")
ax.axvline(0, color="black", linestyle="--", linewidth=1, label="No relationship")
ax.axvline(low, color="#8C3B2E", linestyle=":", label="95% interval")
ax.axvline(high, color="#8C3B2E", linestyle=":")
ax.set_title("Bootstrap Distribution of Sentiment vs. Return Correlation")
ax.set_xlabel("Correlation (r)")
ax.legend()
plt.tight_layout()
plt.savefig("reports/figures/bootstrap_correlation.png", dpi=150)
plt.close()
print("\nSaved outputs to data/processed/ and reports/figures/")
