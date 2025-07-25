# Full app code would be here. It will be injected from the live canvas code.
# RedRisk AI: Main Streamlit App (complete scaffold with Reddit + GARCH)

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import praw
from transformers import pipeline
from arch import arch_model

# --- API KEYS ---
REDDIT_CLIENT_ID = st.secrets["REDDIT_CLIENT_ID"]
REDDIT_SECRET = st.secrets["REDDIT_SECRET"]
REDDIT_USER_AGENT = st.secrets["REDDIT_USER_AGENT"]

# --- SETUP ---
st.set_page_config(page_title="RedRisk AI", layout="wide")
st.title("📊 RedRisk AI – Real-Time Reddit Sentiment & Financial Risk Analysis")

# --- SIDEBAR ---
st.sidebar.header("User Controls")
ticker = st.sidebar.text_input("Enter a stock ticker (e.g. AAPL, TSLA):", value="TSLA")
days_back = st.sidebar.slider("How many days of data?", 30, 365, 180)

# --- PRICE DATA ---
def get_price_data(ticker, days=180):
    end = datetime.today()
    start = end - timedelta(days=days)
    data = yf.download(ticker, start=start, end=end)
    return data

# --- REDDIT SETUP ---
def get_reddit_comments(ticker):
    reddit = praw.Reddit(
        client_id=REDDIT_CLIENT_ID,
        client_secret=REDDIT_SECRET,
        user_agent=REDDIT_USER_AGENT
    )
    posts = reddit.subreddit("stocks").search(ticker, limit=50)
    comments = []
    for post in posts:
        post.comments.replace_more(limit=0)
        comments.extend([comment.body for comment in post.comments.list() if comment.body])
    return comments[:100]

@st.cache_data
def analyze_sentiment(texts):
    classifier = pipeline("sentiment-analysis", model="ProsusAI/finbert")
    return classifier(texts)

# --- GARCH MODELING ---
def run_garch(price_series):
    returns = 100 * price_series.pct_change().dropna()
    model = arch_model(returns, vol='Garch', p=1, q=1)
    res = model.fit(disp="off")
    forecast = res.forecast(horizon=5)
    return res, forecast

# --- PRICE CHART ---
with st.spinner("Fetching stock price data..."):
    price_data = get_price_data(ticker, days_back)
    if isinstance(price_data.columns, pd.MultiIndex):
        price_data.columns = price_data.columns.get_level_values(0)

if 'Close' not in price_data.columns:
    st.error("No 'Close' column found in price data. Try a different ticker.")
    st.stop()

st.line_chart(price_data['Close'], use_container_width=True)

# --- REDDIT + SENTIMENT ---
col1, col2 = st.columns(2)
with col1:
    st.subheader("🧠 Reddit Sentiment Summary")
    try:
        comments = get_reddit_comments(ticker)
        if comments:
            sentiments = analyze_sentiment(comments)
            df_sent = pd.DataFrame(sentiments)
            score = df_sent['label'].value_counts(normalize=True).to_dict()
            st.write("Sentiment breakdown:", score)
            st.bar_chart(df_sent['label'].value_counts())
        else:
            st.info("No relevant Reddit comments found.")
    except Exception as e:
        st.warning(f"Reddit API Error: {e}")

# --- GARCH MODEL ---
with col2:
    st.subheader("📉 GARCH Risk Modeling")
    try:
        res, forecast = run_garch(price_data['Close'])
        st.write("Last 5-day volatility forecast:", forecast.variance.iloc[-1].values)
        st.line_chart(res.conditional_volatility)
    except Exception as e:
        st.warning(f"GARCH Error: {e}")

# --- AI COPILOT (Transformer-based Local) ---
st.subheader("🤖 AI Copilot Advice")
question = st.text_input("Ask a question about this stock:", value=f"Should I buy {ticker}?")

if question:
    sentiment_text = f"Reddit sentiment: {score}" if 'score' in locals() else "Sentiment data unavailable."
    vol_text = f"Volatility: {forecast.variance.iloc[-1].values}" if 'forecast' in locals() else "Volatility data unavailable."

    try:
        with st.spinner("Generating AI response..."):
            qa_pipeline = pipeline("text2text-generation", model="google/flan-t5-base")
            full_prompt = f"{question}. {sentiment_text} {vol_text}"
            response = qa_pipeline(full_prompt)
            st.success(response[0]['generated_text'])
    except Exception as e:
        st.warning(f"AI Error: {e}")
