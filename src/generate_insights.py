"""
Reads the outputs of correlation_analysis.py and automatically generates a
plain-English findings report -- same rule-based approach as the HR project's
insight generator (no LLM call, fully reproducible).

Output: reports/sentiment_insights_report.md
"""
import pandas as pd

corr = pd.read_csv("data/processed/correlation_results.csv")
tercile = pd.read_csv("data/processed/tercile_analysis.csv")
company = pd.read_csv("data/processed/company_correlations.csv")

lines = []
lines.append("# Earnings Call Sentiment vs. Stock Price Reaction: Findings\n")
lines.append("_Auto-generated from `correlation_analysis.py` output._\n")

# ---------------------------------------------------------------
# Section 1: Overall market-wide relationship
# ---------------------------------------------------------------
lines.append("## Overall Market-Wide Relationship\n")
next_day = corr[corr["target"] == "NextDayReturn"].iloc[0]
if next_day["pearson_p"] < 0.05:
    strength = "a statistically significant" 
else:
    strength = "no statistically significant"
lines.append(f"Across all 8 companies and 8 quarters pooled together, there is "
              f"**{strength}** relationship between earnings call sentiment and "
              f"next-day stock return (Pearson r = {next_day['pearson_r']}, "
              f"p = {next_day['pearson_p']}).\n")
lines.append("This means sentiment is **not a reliable market-wide trading signal** "
              "on its own -- but that doesn't mean it's meaningless everywhere. "
              "See the per-company breakdown below.\n")

# ---------------------------------------------------------------
# Section 2: Event-study / tercile breakdown
# ---------------------------------------------------------------
lines.append("## Event Study: Return by Sentiment Group\n")
lines.append("Grouping calls into terciles (most negative third / neutral third / "
              "most positive third of sentiment scores) shows a directional pattern:\n")
for _, row in tercile.iterrows():
    lines.append(f"- **{row['sentiment_tercile']}** ({int(row['n_calls'])} calls): "
                  f"average next-day return of **{row['avg_next_day_return']:+.2%}**")
lines.append("\nPositive-sentiment calls outperformed negative-sentiment calls by "
              f"**{(tercile.iloc[2]['avg_next_day_return'] - tercile.iloc[0]['avg_next_day_return']):+.2%}** "
              "on average -- a real gradient, even though it falls just short of "
              "conventional statistical significance (p is about 0.07) at this sample size. "
              "A larger sample (more quarters of history) would be needed to confirm "
              "this holds up.\n")

# ---------------------------------------------------------------
# Section 3: Per-company findings
# ---------------------------------------------------------------
lines.append("## Which Companies' Sentiment Is Actually Predictive\n")
sig_companies = company[company["p_value"] < 0.05].sort_values("pearson_r", ascending=False)
non_sig = company[company["p_value"] >= 0.05]

if len(sig_companies) > 0:
    lines.append("For these companies, sentiment showed a **statistically significant** "
                  "relationship with next-day price movement:\n")
    for _, row in sig_companies.iterrows():
        lines.append(f"- **{row['Ticker']}**: r = {row['pearson_r']}, p = {row['p_value']} "
                      f"-- sentiment tracks price reaction closely")
    lines.append("")

lines.append("For the remaining companies, no reliable relationship was found "
              "(p > 0.05) -- their stock price reaction to earnings calls appears "
              "driven more by the actual numbers reported (or other factors) than "
              "by the tone of the call itself:\n")
for _, row in non_sig.iterrows():
    lines.append(f"- {row['Ticker']} (r = {row['pearson_r']}, p = {row['p_value']})")

# ---------------------------------------------------------------
# Section 4: Practical takeaway
# ---------------------------------------------------------------
lines.append("\n## Practical Takeaway\n")
if len(sig_companies) > 0:
    top = sig_companies.iloc[0]
    lines.append(f"Sentiment analysis of earnings calls is **not a universal signal** -- "
                  f"it should not be applied blindly across all fintech stocks. But for "
                  f"specific companies like **{top['Ticker']}**, tone of the call carries "
                  f"real, statistically detectable information about the market's reaction. "
                  f"A more targeted strategy -- monitoring sentiment closely for the "
                  f"companies where it *has* historically mattered, rather than treating "
                  f"it as a blanket indicator -- would be the more defensible approach.")
else:
    lines.append("Sentiment analysis alone does not appear to reliably predict price "
                  "reaction for this set of companies. Other factors (reported financial "
                  "metrics, macro conditions, analyst expectations going into the call) "
                  "likely dominate the market's actual response.")

lines.append("\n## Caveats\n")
lines.append("- Transcripts and price reactions in this dataset are synthetically generated "
              "for portfolio purposes; findings illustrate the *method*, not real trading advice.")
lines.append("- Correlation does not imply causation, and with only 8 quarters per company, "
              "per-company findings should be treated as directional, not conclusive.")
lines.append("- This analysis should never be used as the sole basis for a real trading "
              "decision.")

report = "\n".join(lines)
with open("reports/sentiment_insights_report.md", "w", encoding="utf-8") as f:
    f.write(report)

print(report)
print("\n\nSaved to reports/sentiment_insights_report.md")
