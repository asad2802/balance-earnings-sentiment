"""
Generates realistic synthetic earnings call transcripts for 8 fintech/banking
companies across 8 quarters each, with a genuine (controlled) relationship
between transcript sentiment and the stock's next-day price reaction.

Why synthetic: real full-text earnings call transcripts are mostly paywalled.
This generator builds transcripts from a bank of realistic finance-speak
sentence fragments (positive / neutral / cautious / negative), so the
resulting sentiment scores are genuinely extractable by an NLP sentiment
model, and the underlying price reaction is generated FROM the sentiment
(plus noise) so later correlation analysis has something real to recover --
same design principle as the HR attrition project's synthetic data.

Companies:
  JPM (JPMorgan Chase), GS (Goldman Sachs), BAC (Bank of America),
  MS (Morgan Stanley), WFC (Wells Fargo), C (Citigroup),
  PYPL (PayPal), V (Visa)

Output: data/raw/earnings_transcripts.csv
        data/raw/price_reactions.csv   (synthetic -- see note in file)
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

np.random.seed(42)

COMPANIES = {
    "JPM": "JPMorgan Chase", "GS": "Goldman Sachs", "BAC": "Bank of America",
    "MS": "Morgan Stanley", "WFC": "Wells Fargo", "C": "Citigroup",
    "PYPL": "PayPal", "V": "Visa",
}

QUARTERS = ["Q1 2024", "Q2 2024", "Q3 2024", "Q4 2024", "Q1 2025", "Q2 2025",
            "Q3 2025", "Q4 2025"]

# ---------------------------------------------------------------
# Sentence banks by tone -- these get sampled to build each transcript.
# A transcript's "true" sentiment (used to drive price reaction) is the
# proportion of positive vs cautious/negative sentences it's built from.
# ---------------------------------------------------------------
POSITIVE = [
    "We delivered record revenue this quarter, well above consensus estimates.",
    "Net interest income grew meaningfully as we continued to gain market share.",
    "Our trading and investment banking divisions both posted their best results in years.",
    "Credit quality remains strong, with delinquencies at historic lows.",
    "We are raising our full-year guidance given the strength we're seeing across segments.",
    "Client deposits grew substantially, reflecting continued confidence in our platform.",
    "We returned significant capital to shareholders through buybacks and dividend increases.",
    "Digital engagement is accelerating faster than we projected at the start of the year.",
    "Expense discipline combined with revenue growth drove strong operating leverage.",
    "We are extremely pleased with the momentum across every major business line.",
]

NEUTRAL = [
    "Results this quarter were largely in line with our prior guidance.",
    "We saw stable performance across most business segments.",
    "Overall trends remained consistent with what we communicated last quarter.",
    "Net income was roughly flat year over year, as expected.",
    "We continue to monitor the macroeconomic environment closely.",
    "Our balance sheet remains well positioned heading into next quarter.",
    "We are maintaining our current guidance range for the full year.",
]

CAUTIOUS = [
    "We are seeing some softness in consumer spending that bears watching.",
    "Net interest margin compressed slightly due to a competitive deposit environment.",
    "We've increased loan loss provisions given some early signs of credit normalization.",
    "Trading revenue was more volatile than usual given market conditions this quarter.",
    "We are being more conservative in our outlook given macro uncertainty.",
    "Expenses came in slightly higher than anticipated due to continued technology investment.",
]

NEGATIVE = [
    "Revenue fell short of expectations, driven by weaker-than-expected trading results.",
    "We are lowering our full-year guidance given a more challenging environment.",
    "Credit losses increased meaningfully compared to the prior quarter.",
    "We saw meaningful deposit outflows as clients sought higher-yielding alternatives.",
    "Regulatory and legal costs weighed heavily on results this quarter.",
    "We are pausing buybacks to preserve capital given the uncertain outlook.",
    "Management acknowledged the quarter fell well below internal targets.",
]

TONE_BANKS = {"positive": POSITIVE, "neutral": NEUTRAL,
              "cautious": CAUTIOUS, "negative": NEGATIVE}

# Each company gets a slight "personality" (baseline tone tendency) so results
# aren't uniformly random across companies -- e.g. some run consistently
# stronger than others, which is realistic and gives the dashboard more
# interesting company-level patterns to surface.
COMPANY_BIAS = {
    "JPM": 0.25, "GS": 0.10, "BAC": 0.00, "MS": 0.05,
    "WFC": -0.10, "C": -0.15, "PYPL": -0.05, "V": 0.20,
}

rows_transcripts = []
rows_prices = []

base_date = datetime(2024, 1, 20)

for ticker, name in COMPANIES.items():
    for qi, quarter in enumerate(QUARTERS):
        # true underlying sentiment for this quarter's call, in [-1, 1]
        true_sentiment = np.clip(
            np.random.normal(loc=COMPANY_BIAS[ticker], scale=0.55), -1, 1)

        # translate true_sentiment into a mix of tone-bank sampling weights
        if true_sentiment > 0.4:
            weights = {"positive": 0.6, "neutral": 0.3, "cautious": 0.1, "negative": 0.0}
        elif true_sentiment > 0.1:
            weights = {"positive": 0.4, "neutral": 0.4, "cautious": 0.2, "negative": 0.0}
        elif true_sentiment > -0.1:
            weights = {"positive": 0.2, "neutral": 0.5, "cautious": 0.25, "negative": 0.05}
        elif true_sentiment > -0.4:
            weights = {"positive": 0.1, "neutral": 0.3, "cautious": 0.4, "negative": 0.2}
        else:
            weights = {"positive": 0.05, "neutral": 0.15, "cautious": 0.3, "negative": 0.5}

        n_sentences = np.random.randint(9, 13)
        tones = np.random.choice(list(weights.keys()), size=n_sentences,
                                  p=list(weights.values()))
        sentences = [np.random.choice(TONE_BANKS[t]) for t in tones]
        transcript = (f"{name} ({ticker}) {quarter} Earnings Call. "
                      + " ".join(sentences))

        earnings_date = base_date + timedelta(days=90 * qi + np.random.randint(-3, 3))

        rows_transcripts.append({
            "Ticker": ticker, "Company": name, "Quarter": quarter,
            "EarningsDate": earnings_date.strftime("%Y-%m-%d"),
            "Transcript": transcript,
            "_true_sentiment": round(true_sentiment, 3),  # hidden ground truth, for validation only
        })

        # ---- Generate the price reaction FROM sentiment + noise ----
        # This mirrors the HR project's approach: the "true" relationship is
        # baked in, so later correlation analysis has something genuine to find.
        next_day_return = np.clip(
            0.015 * true_sentiment + np.random.normal(0, 0.018), -0.12, 0.12)
        three_day_return = np.clip(
            next_day_return * 1.4 + np.random.normal(0, 0.012), -0.15, 0.15)

        rows_prices.append({
            "Ticker": ticker, "Quarter": quarter,
            "EarningsDate": earnings_date.strftime("%Y-%m-%d"),
            "NextDayReturn": round(next_day_return, 4),
            "ThreeDayReturn": round(three_day_return, 4),
        })

transcripts_df = pd.DataFrame(rows_transcripts)
prices_df = pd.DataFrame(rows_prices)

transcripts_df.to_csv("data/raw/earnings_transcripts.csv", index=False)
prices_df.to_csv("data/raw/price_reactions.csv", index=False)

print(f"Generated {len(transcripts_df)} transcripts across {len(COMPANIES)} companies "
      f"and {len(QUARTERS)} quarters.")
print(f"Saved data/raw/earnings_transcripts.csv and data/raw/price_reactions.csv")
print("\nSample transcript:")
print(transcripts_df.iloc[0]["Transcript"])
print(f"\n(hidden true sentiment for validation: {transcripts_df.iloc[0]['_true_sentiment']})")
