"""
Scores each earnings call transcript's sentiment using VADER, validates the
scores against the hidden ground-truth sentiment (to confirm the NLP model is
actually picking up the right signal), and merges with price reaction data
into one analysis-ready dataset.

Output: data/processed/sentiment_scored.csv
"""
import pandas as pd
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from scipy.stats import pearsonr

analyzer = SentimentIntensityAnalyzer()

print("=== Loading raw data ===")
transcripts = pd.read_csv("data/raw/earnings_transcripts.csv")
prices = pd.read_csv("data/raw/price_reactions.csv")
print(f"Loaded {len(transcripts)} transcripts, {len(prices)} price records")

# ---------------------------------------------------------------
# 1. Score sentiment for each transcript
# ---------------------------------------------------------------
print("\n=== Scoring sentiment with VADER ===")

def score_transcript(text):
    scores = analyzer.polarity_scores(text)
    return pd.Series({
        "sentiment_compound": scores["compound"],   # overall score, -1 to +1
        "sentiment_pos": scores["pos"],              # proportion positive
        "sentiment_neu": scores["neu"],               # proportion neutral
        "sentiment_neg": scores["neg"],               # proportion negative
    })

sentiment_scores = transcripts["Transcript"].apply(score_transcript)
transcripts = pd.concat([transcripts, sentiment_scores], axis=1)

# ---------------------------------------------------------------
# 2. Validate VADER against the hidden ground-truth sentiment
#    (this is the "did my NLP model actually work" check)
# ---------------------------------------------------------------
print("\n=== Validating VADER against ground truth ===")
r, p = pearsonr(transcripts["_true_sentiment"], transcripts["sentiment_compound"])
print(f"Correlation between VADER score and true sentiment: r = {r:.3f}, p = {p:.4f}")
if r > 0.6:
    print("-> Strong agreement: VADER is reliably recovering the underlying sentiment signal.")
elif r > 0.3:
    print("-> Moderate agreement: VADER captures the signal reasonably well, with some noise.")
else:
    print("-> Weak agreement: consider a finance-tuned model (e.g. FinBERT) instead.")

# ---------------------------------------------------------------
# 3. Merge with price reaction data
# ---------------------------------------------------------------
print("\n=== Merging with price reaction data ===")
merged = transcripts.merge(prices[["Ticker", "Quarter", "NextDayReturn", "ThreeDayReturn"]],
                            on=["Ticker", "Quarter"], how="left")

merged = merged.drop(columns=["_true_sentiment"])  # ground truth was for validation only

merged.to_csv("data/processed/sentiment_scored.csv", index=False)
print(f"\nSaved data/processed/sentiment_scored.csv -- {merged.shape}")

print("\n=== Sample scored rows ===")
print(merged[["Ticker", "Quarter", "sentiment_compound", "NextDayReturn"]].head(10).to_string())

print("\n=== Sentiment summary by company ===")
print(merged.groupby("Ticker")["sentiment_compound"].agg(["mean", "std"]).round(3)
      .sort_values("mean", ascending=False))
