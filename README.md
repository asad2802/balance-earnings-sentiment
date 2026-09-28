# Fintech Earnings Sentiment vs. Stock Price Reaction
https://sentimentxls.streamlit.app/

**Business problem:** Does market reaction to earnings calls align with the sentiment
of what executives actually said? This project scores earnings call sentiment for
8 major fintech/banking companies and tests whether it's a leading indicator of the
stock's next-day price move.

**Companies:** JPMorgan (JPM), Goldman Sachs (GS), Bank of America (BAC),
Morgan Stanley (MS), Wells Fargo (WFC), Citigroup (C), PayPal (PYPL), Visa (V)

## Project structure
- `data/raw/` — earnings transcripts + price reactions (synthetic, see note below)
- `data/processed/` — sentiment-scored, merged analysis dataset
- `src/` — data generation, sentiment scoring, statistical analysis
- `app/` — Streamlit dashboard
- `models/`, `reports/` — saved outputs

## Setup
```
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Run the pipeline
```
python src/generate_data.py
python src/sentiment_scoring.py     # (next step)
python src/correlation_analysis.py  # (next step)
```

## Run the app
```
streamlit run app/app.py
```

## A note on data
Transcripts here are realistically generated (not scraped, since full transcripts
are mostly paywalled) with a controlled ground-truth sentiment that drives the
synthetic price reaction — this lets the full pipeline be built and validated
end-to-end. `src/fetch_real_prices.py` (real yfinance data) and instructions for
pulling real transcripts via Alpha Vantage's free API are included so real data
can be swapped in.

## Status
- [x] Synthetic transcript + price reaction generation
- [ ] Sentiment scoring (VADER)
- [ ] Correlation / event-study analysis
- [ ] Auto-generated insights
- [ ] Streamlit dashboard
- [ ] Deployment
