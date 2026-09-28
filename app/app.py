"""
BALANCE - Earnings Call Sentiment vs. Stock Price Reaction Dashboard

Run with: streamlit run app/app.py  (from the project root)

Tabs:
  1. Browse calls  - pick a company/quarter, read the transcript, see sentiment
                      breakdown and the actual price reaction (balance-beam viz)
  2. Insights       - charts + auto-generated findings report
  3. About          - methodology, caveats
"""
import base64
import json
import re

import pandas as pd
import streamlit as st
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

st.set_page_config(page_title="BALANCE · Earnings Sentiment", page_icon="assets/logo-icon.png", layout="wide")

# ---------------------------------------------------------------
# Brand styling — BALANCE: spreadsheet-inspired design system
# ---------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@500;600;700&family=IBM+Plex+Mono:wght@500;600&display=swap');

:root {
  --ribbon: #217346;
  --ribbon-dark: #185C37;
  --grid: #D4D4D4;
  --paper: #FAFAF8;
  --ink: #262626;
  --cell-pos-bg: #C6EFCE; --cell-pos-text: #006100;
  --cell-neg-bg: #FFC7CE; --cell-neg-text: #9C0006;
}

html, body, [class*="css"] { font-family: 'Inter', sans-serif; color: var(--ink); }
.stApp {
  background-color: var(--paper);
  background-image:
    linear-gradient(var(--grid) 1px, transparent 1px),
    linear-gradient(90deg, var(--grid) 1px, transparent 1px);
  background-size: 28px 28px;
  background-position: -1px -1px;
}

/* Fake app title bar -- reads as software, not a logo/masthead */
.title-bar {
  display: flex; align-items: center; gap: 10px;
  background: var(--ribbon-dark); margin: -1rem -1rem 0 -1rem;
  padding: 8px 18px; color: #E8F2EC;
}
.title-dots { display: flex; gap: 6px; }
.title-dot { width: 11px; height: 11px; border-radius: 50%; display: inline-block; }
.title-dot.red { background: #E5544D; }
.title-dot.yellow { background: #E5B84D; }
.title-dot.green { background: #3FBF6F; }
.title-logo { width: 26px; height: 26px; border-radius: 6px; margin-left: 10px; }
.title-brand {
  font-family: 'Inter', sans-serif; font-weight: 700; font-size: 1rem; letter-spacing: 0.02em;
  color: #FFFFFF;
}
.title-filename {
  font-family: 'IBM Plex Mono', monospace; font-size: 0.82rem; opacity: 0.9; margin-left: -6px;
}

/* Formula bar */
.formula-bar {
  display: flex; align-items: center; gap: 0; margin-top: 0;
  border: 1px solid var(--grid); border-top: none; background: white; overflow: hidden;
}
.name-box {
  font-family: 'IBM Plex Mono', monospace; font-weight: 600; font-size: 0.95rem;
  color: var(--ribbon-dark); background: #F0F0F0; border-right: 1px solid var(--grid);
  padding: 8px 16px; min-width: 80px;
}
.fx-icon {
  font-family: 'IBM Plex Mono', monospace; font-style: italic; color: #6B6558;
  border-right: 1px solid var(--grid); padding: 8px 12px;
}
.formula-text { padding: 8px 14px; font-family: 'IBM Plex Mono', monospace; font-size: 0.9rem; color: var(--ink); }

/* Column-letter ruler, purely decorative, reinforces the grid concept */
.col-ruler {
  display: flex; border: 1px solid var(--grid); border-top: none; background: #F5F5F3;
  font-family: 'IBM Plex Mono', monospace; font-size: 0.72rem; color: #9A9584;
}
.col-ruler span { flex: 1; text-align: center; padding: 3px 0; border-right: 1px solid var(--grid); }
.col-ruler span:last-child { border-right: none; }

h2, h3 { font-family: 'Inter', sans-serif !important; font-weight: 700 !important; color: var(--ink) !important; }

/* Sheet tabs -- styled like real spreadsheet tabs at the bottom of a workbook */
.stTabs [data-baseweb="tab-list"] {
  gap: 3px; border-bottom: 2px solid var(--ribbon); padding-top: 6px;
}
button[data-baseweb="tab"] {
  font-family: 'Inter', sans-serif; font-weight: 500; font-size: 0.88rem; color: #6B6558;
  background: #E4E0D8 !important; border: 1px solid var(--grid) !important; border-bottom: none !important;
  border-radius: 6px 6px 0 0 !important; padding: 6px 18px !important;
}
button[data-baseweb="tab"][aria-selected="true"] {
  color: var(--ribbon-dark) !important; background: white !important; font-weight: 700 !important;
  border-bottom: 2px solid white !important; margin-bottom: -2px;
}

.stButton > button[kind="primary"] {
  background-color: var(--ribbon); border: none; border-radius: 3px;
  font-weight: 600; letter-spacing: 0.01em;
}
.stButton > button[kind="primary"]:hover { background-color: var(--ribbon-dark); }

/* Conditional-formatting cell */
.cell-card { background: white; border: 1px solid var(--grid); border-radius: 3px; padding: 14px 18px; }
.cell-ref {
  font-family: 'IBM Plex Mono', monospace; font-size: 0.75rem; color: #8A8578;
  background: #F0F0F0; display: inline-block; padding: 2px 8px; border-radius: 2px; margin-bottom: 8px;
}
.cell-value {
  font-family: 'IBM Plex Mono', monospace; font-size: 1.7rem; font-weight: 600;
  padding: 6px 12px; border-radius: 2px; display: inline-block;
}
.cell-value.pos { background: var(--cell-pos-bg); color: var(--cell-pos-text); }
.cell-value.neg { background: var(--cell-neg-bg); color: var(--cell-neg-text); }

.transcript-box {
  background: white; border: 1px solid var(--grid); border-left: 4px solid var(--ribbon);
  padding: 18px 22px; font-family: 'Inter', sans-serif; font-size: 0.95rem;
  line-height: 1.9; border-radius: 3px;
}
.sent-pos { background: var(--cell-pos-bg); color: var(--cell-pos-text); padding: 1px 3px; border-radius: 2px; }
.sent-neg { background: var(--cell-neg-bg); color: var(--cell-neg-text); padding: 1px 3px; border-radius: 2px; }
.legend-chip { font-family: 'IBM Plex Mono', monospace; font-size: 0.78rem; padding: 2px 8px; border-radius: 2px; margin-right: 8px; }

/* Scoreboard chips */
.score-row { display: flex; gap: 12px; margin: 12px 0 18px 0; flex-wrap: wrap; }
.score-chip {
  font-family: 'IBM Plex Mono', monospace; font-weight: 600; font-size: 0.95rem;
  padding: 8px 16px; border-radius: 4px; border: 1px solid var(--grid);
}
.score-chip.pos { background: var(--cell-pos-bg); color: var(--cell-pos-text); }
.score-chip.neg { background: var(--cell-neg-bg); color: var(--cell-neg-text); }
.score-chip.neu { background: #F0F0F0; color: #6B6558; }

/* Grouped sentence boxes */
.group-card { background: white; border: 1px solid var(--grid); border-radius: 4px; margin-bottom: 16px; overflow: hidden; }
.group-header {
  font-family: 'Inter', sans-serif; font-weight: 700; font-size: 1rem;
  padding: 12px 18px; display: flex; align-items: center; justify-content: space-between;
}
.group-header.pos { background: var(--cell-pos-bg); color: var(--cell-pos-text); }
.group-header.neg { background: var(--cell-neg-bg); color: var(--cell-neg-text); }
.group-header.neu { background: #ECECEC; color: #4A463C; }
.group-count {
  font-family: 'IBM Plex Mono', monospace; font-size: 0.8rem; background: rgba(255,255,255,0.6);
  padding: 2px 10px; border-radius: 10px;
}
.bullet-list { list-style: none; margin: 0; padding: 6px 18px 14px 18px; }
.bullet-list li {
  padding: 8px 0 8px 22px; position: relative; border-bottom: 1px dashed var(--grid);
  font-family: 'Inter', sans-serif; font-size: 0.92rem; line-height: 1.5;
}
.bullet-list li:last-child { border-bottom: none; }
.bullet-list li::before { content: "▸"; position: absolute; left: 0; color: #8A8578; }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data():
    df = pd.read_csv("data/processed/sentiment_scored.csv")
    return df


@st.cache_data
def load_correlation_results():
    return pd.read_csv("data/processed/correlation_results.csv")


@st.cache_data
def load_tercile():
    return pd.read_csv("data/processed/tercile_analysis.csv")


@st.cache_data
def load_company_corr():
    return pd.read_csv("data/processed/company_correlations.csv")


@st.cache_data
def load_insights_report():
    with open("reports/sentiment_insights_report.md", encoding="utf-8") as f:
        return f.read()


@st.cache_resource
def load_price_model():
    with open("models/sentiment_price_model.json") as f:
        return json.load(f)


@st.cache_resource
def get_analyzer():
    return SentimentIntensityAnalyzer()


@st.cache_resource
def get_logo_base64():
    with open("assets/logo-icon.png", "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def split_sentences(text):
    # Simple, dependency-free sentence splitter (avoids needing an nltk data
    # download at runtime, which can be unreliable on a fresh deploy).
    text = text.strip()
    parts = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
    return [p.strip() for p in parts if p.strip()]


def highlight_transcript(text, analyzer, neutral_band=0.05):
    """Returns (html, overall_compound, sentence_scores) with each sentence
    tinted using Excel's conditional-formatting colors based on its own
    VADER compound score."""
    sentences = split_sentences(text)
    html_parts = []
    sentence_scores = []
    for s in sentences:
        score = analyzer.polarity_scores(s)["compound"]
        sentence_scores.append((s, score))
        if score > neutral_band:
            html_parts.append(f'<span class="sent-pos">{s}</span>')
        elif score < -neutral_band:
            html_parts.append(f'<span class="sent-neg">{s}</span>')
        else:
            html_parts.append(s)
    overall = analyzer.polarity_scores(text)["compound"]
    return " ".join(html_parts), overall, sentence_scores


def render_grouped_transcript(text, analyzer, neutral_band=0.05):
    """Renders a transcript as a scoreboard (positive/neutral/negative counts)
    plus three bulleted boxes -- much easier to skim than one highlighted
    wall of text. Returns the overall compound score for reuse elsewhere."""
    sentences = split_sentences(text)
    pos_list, neu_list, neg_list = [], [], []
    for s in sentences:
        score = analyzer.polarity_scores(s)["compound"]
        if score > neutral_band:
            pos_list.append(s)
        elif score < -neutral_band:
            neg_list.append(s)
        else:
            neu_list.append(s)
    overall = analyzer.polarity_scores(text)["compound"]

    st.markdown(f"""
    <div class="score-row">
      <div class="score-chip pos">🟢 {len(pos_list)} strengths</div>
      <div class="score-chip neg">🔴 {len(neg_list)} concerns</div>
      <div class="score-chip neu">⚪ {len(neu_list)} neutral notes</div>
    </div>
    """, unsafe_allow_html=True)

    def render_box(title, icon, items, css_class):
        if not items:
            return
        bullets = "".join(f"<li>{s}</li>" for s in items)
        st.markdown(f"""
        <div class="group-card">
          <div class="group-header {css_class}">
            <span>{icon} {title}</span>
            <span class="group-count">{len(items)}</span>
          </div>
          <ul class="bullet-list">{bullets}</ul>
        </div>
        """, unsafe_allow_html=True)

    render_box("Strengths called out", "🟢", pos_list, "pos")
    render_box("Concerns or caution flagged", "🔴", neg_list, "neg")
    with st.expander(f"⚪ Neutral / factual statements ({len(neu_list)})"):
        for s in neu_list:
            st.write(f"▸ {s}")

    return overall


df = load_data()
price_model = load_price_model()
analyzer = get_analyzer()

logo_b64 = get_logo_base64()
st.markdown(f"""
<div class="title-bar">
  <div class="title-dots">
    <span class="title-dot red"></span>
    <span class="title-dot yellow"></span>
    <span class="title-dot green"></span>
  </div>
  <img class="title-logo" src="data:image/png;base64,{logo_b64}" alt="BALANCE logo">
  <span class="title-brand">BALANCE</span>
  <span class="title-filename">.xlsx — Earnings Sentiment Workbook</span>
</div>
<div class="formula-bar">
  <div class="name-box">A1</div>
  <div class="fx-icon">fx</div>
  <div class="formula-text">="what they said, weighed against what it moved"</div>
</div>
<div class="col-ruler">
  <span>A</span><span>B</span><span>C</span><span>D</span><span>E</span>
  <span>F</span><span>G</span><span>H</span><span>I</span><span>J</span>
</div>
<br>
""", unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs(["Browse calls", "Analyze a transcript", "Insights", "About"])

# =================================================================
# TAB 1: BROWSE CALLS
# =================================================================
with tab1:
    st.subheader("Pick a call")
    col_a, col_b = st.columns(2)
    with col_a:
        ticker = st.selectbox("Company", sorted(df["Ticker"].unique()))
    with col_b:
        quarters_available = df[df["Ticker"] == ticker]["Quarter"].tolist()
        quarter = st.selectbox("Quarter", quarters_available)

    row = df[(df["Ticker"] == ticker) & (df["Quarter"] == quarter)].iloc[0]

    st.markdown(f"### {row['Company']} ({row['Ticker']}) — {row['Quarter']}")
    st.caption(f"Earnings date: {row['EarningsDate']}")

    render_grouped_transcript(row["Transcript"], analyzer)

    st.divider()
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Call sentiment**")
        sentiment = row["sentiment_compound"]
        direction = "pos" if sentiment >= 0 else "neg"
        st.markdown(f"""
        <div class="cell-card">
          <div class="cell-ref">C4</div><br>
          <span class="cell-value {direction}">{sentiment:+.3f}</span>
        </div>
        """, unsafe_allow_html=True)
        st.caption(f"Positive: {row['sentiment_pos']:.0%} · Neutral: {row['sentiment_neu']:.0%} "
                   f"· Negative: {row['sentiment_neg']:.0%} of sentence tone")

    with col2:
        st.markdown("**Actual price reaction**")
        ret = row["NextDayReturn"]
        direction = "pos" if ret >= 0 else "neg"
        st.markdown(f"""
        <div class="cell-card">
          <div class="cell-ref">D4</div><br>
          <span class="cell-value {direction}">{ret:+.2%}</span>
        </div>
        """, unsafe_allow_html=True)
        st.caption(f"3-day return: {row['ThreeDayReturn']:+.2%}")

# =================================================================
# TAB 2: ANALYZE A TRANSCRIPT (live input -> live output)
# =================================================================
with tab2:
    st.subheader("Paste or upload an earnings call transcript")
    st.caption("Works with real transcripts too, not just the demo data above — "
               "paste any earnings call text, or upload a .txt file.")

    company_corr_lookup = load_company_corr().set_index("Ticker")

    col_ctx1, col_ctx2 = st.columns([2, 1])
    with col_ctx1:
        uploaded_file = st.file_uploader("Upload a .txt transcript (optional)", type=["txt"])
        default_text = ""
        if uploaded_file is not None:
            default_text = uploaded_file.read().decode("utf-8", errors="ignore")
        transcript_input = st.text_area(
            "Or paste transcript text here", value=default_text, height=220,
            placeholder="Paste an earnings call transcript, press release, or "
                        "management commentary here...")
    with col_ctx2:
        context_ticker = st.selectbox(
            "Contextualize against which company's history?",
            ["None"] + sorted(df["Ticker"].unique()))
        st.caption("Optional — if selected, the prediction's reliability is "
                   "framed using that company's own historical relationship "
                   "between sentiment and price reaction.")

    analyze_btn = st.button("Analyze", type="primary", use_container_width=True)

    if analyze_btn:
        if not transcript_input.strip():
            st.error("Paste some text or upload a file first.")
        else:
            _, overall_score, sentence_scores = highlight_transcript(
                transcript_input, analyzer)

            st.divider()
            res_col1, res_col2 = st.columns(2)

            with res_col1:
                direction = "pos" if overall_score >= 0 else "neg"
                st.markdown("**Overall sentiment**")
                st.markdown(f"""
                <div class="cell-card">
                  <div class="cell-ref">C4</div><br>
                  <span class="cell-value {direction}">{overall_score:+.3f}</span>
                </div>
                """, unsafe_allow_html=True)

            with res_col2:
                predicted_return = price_model["intercept"] + price_model["slope"] * overall_score
                direction = "pos" if predicted_return >= 0 else "neg"
                st.markdown("**Predicted next-day price reaction**")
                st.markdown(f"""
                <div class="cell-card">
                  <div class="cell-ref">D4</div><br>
                  <span class="cell-value {direction}">{predicted_return:+.2%}</span>
                </div>
                """, unsafe_allow_html=True)
                st.caption(f"Model: predicted return = {price_model['intercept']:.4f} + "
                           f"{price_model['slope']:.4f} × sentiment "
                           f"(overall R² = {price_model['r_squared']:.1%})")

            # Reliability framing based on the selected company's own history
            st.divider()
            if context_ticker != "None" and context_ticker in company_corr_lookup.index:
                row = company_corr_lookup.loc[context_ticker]
                if row["p_value"] < 0.05:
                    st.success(
                        f"For **{context_ticker}**, sentiment has historically been a "
                        f"statistically significant predictor of next-day price reaction "
                        f"(r = {row['pearson_r']}, p = {row['p_value']}). This prediction "
                        f"can be treated with reasonable confidence.")
                else:
                    st.warning(
                        f"For **{context_ticker}**, sentiment has **not** historically been "
                        f"a significant predictor of price reaction (r = {row['pearson_r']}, "
                        f"p = {row['p_value']}). Treat this prediction as a rough estimate only "
                        f"— other factors likely dominate {context_ticker}'s price moves.")
            else:
                st.info("No company selected for context — this prediction uses the "
                        "overall (market-wide) model, which explains only "
                        f"{price_model['r_squared']:.1%} of next-day return variance. "
                        "Select a company above for a more specific reliability read.")

            st.divider()
            st.markdown("**Breakdown**")
            render_grouped_transcript(transcript_input, analyzer)

            top_drivers = sorted(sentence_scores, key=lambda x: abs(x[1]), reverse=True)[:3]
            if top_drivers:
                st.markdown("**Top sentences driving the sentiment score**")
                for s, score in top_drivers:
                    tag = "🟢" if score > 0 else ("🔴" if score < 0 else "⬜")
                    st.write(f"{tag} `{score:+.2f}` — {s}")

# =================================================================
# TAB 3: INSIGHTS
# =================================================================
with tab3:
    st.subheader("Does sentiment predict price reaction?")
    corr = load_correlation_results()
    st.dataframe(corr, use_container_width=True)

    st.divider()
    st.subheader("Event study: return by sentiment group")
    tercile = load_tercile()
    st.bar_chart(tercile.set_index("sentiment_tercile")["avg_next_day_return"])
    st.dataframe(tercile, use_container_width=True)

    st.divider()
    st.subheader("Which companies' sentiment is actually predictive")
    company_corr = load_company_corr()
    company_corr_sorted = company_corr.sort_values("pearson_r", ascending=False)
    st.bar_chart(company_corr_sorted.set_index("Ticker")["pearson_r"])
    st.dataframe(company_corr_sorted, use_container_width=True)

    st.divider()
    st.subheader("Full findings report")
    st.markdown(load_insights_report())

# =================================================================
# TAB 4: ABOUT
# =================================================================
with tab4:
    st.subheader("About BALANCE")
    st.markdown("""
BALANCE weighs what executives said on an earnings call against what the
stock actually did afterward — testing whether sentiment is a leading
indicator of price movement, or just noise.

**Companies covered:** JPMorgan (JPM), Goldman Sachs (GS), Bank of America (BAC),
Morgan Stanley (MS), Wells Fargo (WFC), Citigroup (C), PayPal (PYPL), Visa (V)

**How it works**
1. Scores each earnings call transcript's sentiment using VADER, a lexicon-based
   sentiment analysis tool
2. Validates the sentiment score against known ground truth before trusting it
3. Tests the sentiment-to-price relationship three ways: correlation (Pearson
   and Spearman), linear regression, and an event-study tercile breakdown —
   the same technique real analysts use to isolate a stock's reaction to a
   specific event
4. Breaks the relationship down per company, since a market-wide average can
   hide real signal in individual names
5. Auto-generates a written findings report from the statistical results

**Built with:** Python, pandas, VADER, statsmodels, scipy, Streamlit

**Worth knowing**
- Transcripts and price reactions here are synthetically generated for
  portfolio purposes — not real trading data
- Correlation is not causation, and with only 8 quarters per company,
  per-company findings are directional, not conclusive
- This should never be the sole basis for a real trading decision
""")
