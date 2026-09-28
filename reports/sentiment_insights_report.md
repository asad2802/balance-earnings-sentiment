# Earnings Call Sentiment vs. Stock Price Reaction: Findings

_Auto-generated from `correlation_analysis.py` output._

## Overall Market-Wide Relationship

Across all 8 companies and 8 quarters pooled together, there is **no statistically significant** relationship between earnings call sentiment and next-day stock return (Pearson r = 0.094, p = 0.4598).

This means sentiment is **not a reliable market-wide trading signal** on its own -- but that doesn't mean it's meaningless everywhere. See the per-company breakdown below.

## Event Study: Return by Sentiment Group

Grouping calls into terciles (most negative third / neutral third / most positive third of sentiment scores) shows a directional pattern:

- **Negative/cautious call** (22 calls): average next-day return of **-0.56%**
- **Neutral call** (22 calls): average next-day return of **-0.37%**
- **Positive call** (20 calls): average next-day return of **+0.87%**

Positive-sentiment calls outperformed negative-sentiment calls by **+1.43%** on average -- a real gradient, even though it falls just short of conventional statistical significance (p is about 0.07) at this sample size. A larger sample (more quarters of history) would be needed to confirm this holds up.

## Which Companies' Sentiment Is Actually Predictive

For these companies, sentiment showed a **statistically significant** relationship with next-day price movement:

- **JPM**: r = 0.788, p = 0.0203 -- sentiment tracks price reaction closely
- **PYPL**: r = 0.77, p = 0.0253 -- sentiment tracks price reaction closely

For the remaining companies, no reliable relationship was found (p > 0.05) -- their stock price reaction to earnings calls appears driven more by the actual numbers reported (or other factors) than by the tone of the call itself:

- V (r = 0.654, p = 0.0783)
- BAC (r = 0.311, p = 0.4533)
- C (r = 0.256, p = 0.5407)
- WFC (r = 0.188, p = 0.655)
- MS (r = -0.195, p = 0.6432)
- GS (r = -0.34, p = 0.4101)

## Practical Takeaway

Sentiment analysis of earnings calls is **not a universal signal** -- it should not be applied blindly across all fintech stocks. But for specific companies like **JPM**, tone of the call carries real, statistically detectable information about the market's reaction. A more targeted strategy -- monitoring sentiment closely for the companies where it *has* historically mattered, rather than treating it as a blanket indicator -- would be the more defensible approach.

## Caveats

- Transcripts and price reactions in this dataset are synthetically generated for portfolio purposes; findings illustrate the *method*, not real trading advice.
- Correlation does not imply causation, and with only 8 quarters per company, per-company findings should be treated as directional, not conclusive.
- This analysis should never be used as the sole basis for a real trading decision.