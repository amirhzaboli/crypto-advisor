import streamlit as st
import requests
import yfinance as yf
import ta
import pandas as pd
import numpy as np
import time

# --- Page setup ---
st.set_page_config(page_title="Crypto Portfolio & Buy Advisor", layout="wide")

if "page" not in st.session_state:
    st.session_state.page = "stage1_rules"

if "API_KEY" not in st.session_state:
    st.session_state.API_KEY = ""

if "ACCESS_TOKEN" not in st.session_state:
    st.session_state.ACCESS_TOKEN = ""

# === Shared Helpers ===
def fetch_fear_greed():
    url = "https://api.alternative.me/fng/?limit=1"
    try:
        r = requests.get(url)
        if r.status_code == 200:
            value = int(r.json()["data"][0]["value"])
            if value <= 25: return "Extreme Fear"
            elif value <= 45: return "Fear"
            elif value <= 55: return "Neutral"
            elif value <= 75: return "Greed"
            else: return "Extreme Greed"
        return "Neutral"
    except:
        return "Neutral"

def fetch_news_sentiment(symbol):
    url = f"https://cryptopanic.com/api/v1/posts/?auth_token=demo&currencies={symbol.lower()}"
    try:
        r = requests.get(url)
        if r.status_code == 200:
            posts = r.json().get("results", [])
            sentiments = [p.get("vote", {}).get("value") for p in posts if "vote" in p]
            if sentiments.count("positive") > sentiments.count("negative"):
                return "Positive"
            elif sentiments.count("negative") > sentiments.count("positive"):
                return "Negative"
            else:
                return "Neutral"
    except:
        return "Neutral"
    return "Neutral"

# === Stage 1 Logic ===
def find_nearest_levels(prices, window=5, lookback=30):
    supports, resistances = [], []
    close = prices[-1]
    for i in range(len(prices) - lookback, len(prices) - window):
        segment = prices[i-window:i+window+1]
        if len(segment) < (2*window + 1):
            continue
        if prices[i] == segment.min(): supports.append(prices[i])
        if prices[i] == segment.max(): resistances.append(prices[i])
    stop_loss = max([s for s in supports if s < close], default=None)
    target_sell = min([r for r in resistances if r > close], default=None)
    return stop_loss, target_sell

def analyze_stage1(my_coins):
    results = []
    tickers = [coin + "-AUD" for coin in my_coins]
    fear_greed_status = fetch_fear_greed()

    for ticker in tickers:
        try:
            data = yf.download(ticker, period="3mo", interval="1d", progress=False)
            if data.empty: continue
            close_series = data["Close"].squeeze()
            data["RSI"] = ta.momentum.RSIIndicator(close_series, window=14).rsi()
            macd = ta.trend.MACD(close_series)
            data["MACD"] = macd.macd()
            data["MACD_signal"] = macd.macd_signal()
            data = data.dropna()
            if data.empty: continue

            rsi = float(data["RSI"].iloc[-1])
            macd_val = float(data["MACD"].iloc[-1])
            macd_sig = float(data["MACD_signal"].iloc[-1])
            close = float(data["Close"].iloc[-1])

            signals = []
            if rsi >= 70: signals.append("SELL (RSI overbought)")
            elif rsi <= 30: signals.append("KEEP (RSI oversold)")
            if macd_val > macd_sig: signals.append("KEEP (MACD bullish)")
            elif macd_val < macd_sig: signals.append("SELL (MACD bearish)")

            prices = data["Close"].values
            stop_loss, target_sell = find_nearest_levels(prices)
            coin_symbol = ticker.replace("-AUD", "").lower()
            news_sentiment = fetch_news_sentiment(coin_symbol)

            sell_votes = sum("SELL" in s for s in signals)
            keep_votes = sum("KEEP" in s for s in signals)
            total_votes = max(sell_votes + keep_votes, 1)
            if sell_votes > keep_votes:
                recommendation = f"SELL ({(sell_votes/total_votes)*100:.1f}% confidence)"
            elif keep_votes > sell_votes:
                recommendation = f"KEEP ({(keep_votes/total_votes)*100:.1f}% confidence)"
            else:
                recommendation = "HOLD (Neutral)"

            results.append({
                "Coin": ticker.replace("-AUD",""),
                "Close": round(close, 2),
                "RSI": round(rsi, 2),
                "MACD": round(macd_val, 2),
                "Signal": round(macd_sig, 2),
                "News": news_sentiment,
                "Fear/Greed": fear_greed_status,
                "Stop-Loss": round(float(stop_loss), 2) if stop_loss else None,
                "Target-Sell": round(float(target_sell), 2) if target_sell else None,
                "Recommendation": recommendation
            })
        except:
            continue
    return pd.DataFrame(results)

def fetch_swyftx_balances(api_key, access_token):
    url = "https://api.swyftx.com.au/user/balance/"
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json"
    }
    try:
        resp = requests.get(url, headers=headers)
        if resp.status_code != 200:
            return []
        balances = resp.json()
        if isinstance(balances, list):
            return [item["asset"] for item in balances if float(item.get("available", 0)) > 0]
    except:
        return []
    return []

# === Stage 2 Logic ===
def fetch_top_coins():
    url = "https://api.coinlore.net/api/tickers/?start=0&limit=20"
    r = requests.get(url)
    if r.status_code == 200:
        return [c["symbol"] for c in r.json()["data"]]
    return []

def fetch_candles(symbol, days="90d"):
    ticker = f"{symbol}-USD"
    try:
        data = yf.download(ticker, period=days, interval="1d", progress=False)
        if data.empty: return pd.DataFrame()
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.droplevel(1)
        return data.reset_index()
    except:
        return pd.DataFrame()

def compute_indicators(df):
    close_series = df["Close"]
    df["RSI"] = ta.momentum.RSIIndicator(close_series, window=14).rsi()
    macd = ta.trend.MACD(close_series)
    df["MACD"] = macd.macd()
    df["MACD_signal"] = macd.macd_signal()
    return df.dropna()

def detect_patterns(df):
    patterns, closes = [], df["Close"].values
    if len(closes) < 60: return patterns
    if closes[-1] < closes[-30:-15].max() * 0.95: patterns.append("Double Top (Bearish)")
    if closes[-1] > closes[-30:-15].min() * 1.05: patterns.append("Double Bottom (Bullish)")
    left, head, right = np.mean(closes[-60:-40]), np.max(closes[-40:-20]), np.mean(closes[-20:])
    if head > left * 1.05 and head > right * 1.05: patterns.append("Head & Shoulders (Bearish)")
    if head < left * 0.95 and head < right * 0.95: patterns.append("Inverse H&S (Bullish)")
    if closes[-1] > np.mean(closes[-30:]) * 1.05: patterns.append("Cup & Handle (Bullish)")
    return patterns

def analyze_coin(symbol, trending_coins):
    df = fetch_candles(symbol)
    if df.empty: return None
    df = compute_indicators(df)
    if df.empty: return None
    rsi, macd_val, macd_sig = df["RSI"].iloc[-1], df["MACD"].iloc[-1], df["MACD_signal"].iloc[-1]
    price, vol, vol_med = df["Close"].iloc[-1], df["Volume"].iloc[-1], df["Volume"].median()

    rsi_score = 1 if rsi <= 30 else (-1 if rsi >= 70 else 0)
    macd_score = 1 if macd_val > macd_sig else -1
    vol_score = 1 if vol > 1.5 * vol_med else 0
    patterns = detect_patterns(df)
    pattern_score = sum(1 if "Bullish" in p else -1 for p in patterns)
    news = fetch_news_sentiment(symbol)
    news_score = 1 if news == "Positive" else (-1 if news == "Negative" else 0)
    influencer_score = 1 if symbol.upper() in trending_coins else 0

    weights = {"RSI":0.15, "MACD":0.15, "Volume":0.20, "Pattern":0.25, "News":0.15, "Influencer":0.10}
    raw_score = (
        rsi_score * weights["RSI"] +
        macd_score * weights["MACD"] +
        vol_score * weights["Volume"] +
        pattern_score * weights["Pattern"] +
        news_score * weights["News"] +
        influencer_score * weights["Influencer"]
    )
    confidence = (raw_score + 1) / 2 * 100
    if confidence > 65: recommendation = f"BUY ({confidence:.1f}%)"
    elif confidence >= 40: recommendation = f"HOLD ({confidence:.1f}%)"
    else: recommendation = f"AVOID ({confidence:.1f}%)"

    return {
        "Coin": symbol,
        "Price (USD)": round(price, 4),
        "RSI": round(rsi, 2),
        "MACD": round(macd_val, 4),
        "Signal": round(macd_sig, 4),
        "Volume Spike": "Yes" if vol_score == 1 else "No",
        "Patterns": ", ".join(patterns) if patterns else "None",
        "News": news,
        "Influencer": "Yes" if influencer_score else "No",
        "Recommendation": recommendation
    }

# === UI ===
if st.session_state.page == "stage1_rules":
    st.title("📘 Stage 1: Portfolio Advisor – Rules, Assumptions & Limitations")

    st.markdown("""
    ### 🔹 Purpose
    - This stage analyzes your **current Swyftx portfolio holdings**.
    - It helps you decide whether to **SELL / KEEP / HOLD** each coin.

    ### 🔹 Indicators Used
    1. **RSI** – Overbought (≥70 → SELL), Oversold (≤30 → KEEP)
    2. **MACD** – Bullish (MACD > Signal → KEEP), Bearish (MACD < Signal → SELL)
    3. **Support & Resistance** – Finds recent stop-loss & target-sell
    4. **News Sentiment** – Positive → KEEP, Negative → SELL, Neutral otherwise
    5. **Fear & Greed Index** – Market-wide sentiment measure

    ### 🔹 Confidence Levels
    - Each indicator “votes” for SELL or KEEP
    - Confidence % = (votes ÷ total) × 100
    - Example: RSI=SELL, MACD=SELL → SELL (100% confidence)

    ### 🔹 Limitations
    - Prices from Yahoo Finance (some coins unsupported)
    - News via CryptoPanic demo key (limited coverage)
    - If balances not available → ⚠️ NO COIN IS AVAILABLE
    - Uses last 3 months of daily candles (not intraday)
    - Not financial advice
    """)

    st.session_state.API_KEY = st.text_input("🔑 Swyftx API Key", type="password")
    st.session_state.ACCESS_TOKEN = st.text_input("🪙 Swyftx API Token (JWT)", type="password")

    if st.button("Proceed to Stage 1 Results"):
        st.session_state.page = "stage1_results"

elif st.session_state.page == "stage1_results":
    st.title("📊 Stage 1 Results")

    coins = []
    if st.session_state.API_KEY and st.session_state.ACCESS_TOKEN:
        coins = fetch_swyftx_balances(st.session_state.API_KEY, st.session_state.ACCESS_TOKEN)

    if not coins:
        st.warning("⚠️ NO COIN IS AVAILABLE")
    else:
        df1 = analyze_stage1(coins)
        st.dataframe(df1, use_container_width=True)

    if st.button("Proceed to Stage 2 Rules"):
        st.session_state.page = "stage2_rules"

elif st.session_state.page == "stage2_rules":
    st.title("📘 Stage 2: Buy Suggestor – Rules, Assumptions & Limitations")

    st.markdown("""
    ### 🔹 Purpose
    - This stage analyzes the **Top 20 coins by 24h trading volume**
    - It suggests whether to **BUY / HOLD / AVOID** each coin

    ### 🔹 Indicators & Factors Used
    1. **RSI** – Oversold ≤30 → +1, Overbought ≥70 → -1
    2. **MACD** – Bullish (MACD > Signal) → +1, Bearish → -1
    3. **Volume Spikes** – If volume > 1.5 × median → +1
    4. **Chart Patterns** – Bullish: +1, Bearish: -1
    5. **News Sentiment** – Positive +1, Negative -1
    6. **Influencer Proxy** – If trending on CoinGecko → +1

    ### 🔹 Confidence Levels
    - Weighted system:
        - RSI: 15%, MACD: 15%, Volume: 20%, Pattern: 25%, News: 15%, Influencer: 10%
    - Confidence = (Raw Score + 1)/2 × 100
    - BUY > 65%, HOLD ≥ 40%, AVOID < 40%

    ### 🔹 Limitations
    - Yahoo Finance may not support all coins
    - Pattern detection is simplified
    - News via CryptoPanic demo (limited coverage)
    - Influencer proxy only checks trending coins
    - Uses 90 days of daily candles (not intraday)
    - Not financial advice
    """)

    if st.button("Proceed to Stage 2 Results"):
        st.session_state.page = "stage2_results"

elif st.session_state.page == "stage2_results":
    st.title("📊 Stage 2 Results")

    top_coins = fetch_top_coins()
    trending_coins = []
    try:
        r = requests.get("https://api.coingecko.com/api/v3/search/trending")
        if r.status_code == 200:
            trending_coins = [c["item"]["symbol"].upper() for c in r.json()["coins"]]
    except: pass

    results = []
    for coin in top_coins:
        res = analyze_coin(coin, trending_coins)
        if res: results.append(res)
        time.sleep(1)

    if not results:
        st.warning("⚠️ No analysis generated")
    else:
        df2 = pd.DataFrame(results)
        st.dataframe(df2, use_container_width=True)
