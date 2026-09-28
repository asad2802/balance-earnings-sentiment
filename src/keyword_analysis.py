"""
Keyword analysis: which words most distinguish positive-sounding sentences
from negative-sounding ones across all earnings call transcripts?

Method:
  1. Split every transcript into sentences and score each with VADER
  2. Bucket sentences as positive / negative (neutral ones are ignored here)
  3. Count how often each meaningful word appears in each bucket
  4. Rank words by a simple "lift" score: how much more common the word is
     in one bucket than the other

Output:
  - data/processed/keyword_lift.csv
  - reports/figures/top_keywords.png
"""
import os
import re
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

os.makedirs("reports/figures", exist_ok=True)
analyzer = SentimentIntensityAnalyzer()

STOPWORDS = set("""
a an the and or but of to in on for with at by from as is are was were be been
this that these those it its we our us you your they their than then so if
which who whom what when where while over under into about across both more
most some any each other such very also just not no nor only own same too can
will would should could may might have has had do does did having during
""".split())

df = pd.read_csv("data/processed/sentiment_scored.csv")
print(f"Loaded {len(df)} transcripts")


def split_sentences(text):
    parts = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text.strip())
    return [p.strip() for p in parts if p.strip()]


def tokenize(sentence):
    words = re.findall(r"[a-zA-Z][a-zA-Z\-']+", sentence.lower())
    return [w for w in words if w not in STOPWORDS and len(w) > 2]


pos_counter, neg_counter = Counter(), Counter()
n_pos_sent = n_neg_sent = 0

for transcript in df["Transcript"]:
    for sentence in split_sentences(transcript):
        score = analyzer.polarity_scores(sentence)["compound"]
        if score > 0.05:
            pos_counter.update(set(tokenize(sentence)))
            n_pos_sent += 1
        elif score < -0.05:
            neg_counter.update(set(tokenize(sentence)))
            n_neg_sent += 1

print(f"Positive sentences: {n_pos_sent} | Negative sentences: {n_neg_sent}")

# Lift = (rate in positive bucket + smoothing) / (rate in negative bucket + smoothing)
# log-scaled so the score is symmetric: >0 leans positive, <0 leans negative.
rows = []
vocab = set(pos_counter) | set(neg_counter)
for w in vocab:
    p_rate = (pos_counter[w] + 1) / (n_pos_sent + 2)
    n_rate = (neg_counter[w] + 1) / (n_neg_sent + 2)
    total = pos_counter[w] + neg_counter[w]
    if total >= 3:  # ignore very rare words
        rows.append({"word": w, "in_positive": pos_counter[w],
                      "in_negative": neg_counter[w],
                      "log_lift": round(float(np.log(p_rate / n_rate)), 3)})

kw = pd.DataFrame(rows).sort_values("log_lift", ascending=False)
kw.to_csv("data/processed/keyword_lift.csv", index=False)

print("\n=== Words most associated with POSITIVE sentences ===")
print(kw.head(10).to_string(index=False))
print("\n=== Words most associated with NEGATIVE sentences ===")
print(kw.tail(10).sort_values("log_lift").to_string(index=False))

top = pd.concat([kw.head(8), kw.tail(8)])
fig, ax = plt.subplots(figsize=(8, 6))
colors = ["#2F6D4F" if v > 0 else "#8C3B2E" for v in top["log_lift"]]
ax.barh(top["word"], top["log_lift"], color=colors)
ax.axvline(0, color="black", linewidth=0.8)
ax.set_title("Words That Most Distinguish Positive vs. Negative Call Language")
ax.set_xlabel("Log lift (right = positive tone, left = negative tone)")
ax.invert_yaxis()
plt.tight_layout()
plt.savefig("reports/figures/top_keywords.png", dpi=150)
plt.close()
print("\nSaved data/processed/keyword_lift.csv and reports/figures/top_keywords.png")
