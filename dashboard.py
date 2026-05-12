import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests
import json
import os
from textblob import TextBlob
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="AI Trading Bot", page_icon="🤖", layout="wide")
st.title("🤖 AI Trading Bot Dashboard")
st.markdown("**Strategji bazuar në SMA + RSI + ML Model**")

# Sidebar
st.sidebar.header("⚙️ Cilësimet")
symbol = st.sidebar.selectbox("Zgjidh assetin", ["BTC-USD", "ETH-USD", "AAPL", "TSLA"])
period = st.sidebar.selectbox("Periudha", ["1y", "2y", "5y"])

@st.cache_data
def load_data(symbol, period):
    data = yf.download(symbol, period=period, interval="1d")
    data = pd.DataFrame(data["Close"][symbol])
    data.columns = ["Close"]

    # Indikatort
    data["SMA_20"] = data["Close"].rolling(20).mean()
    data["SMA_50"] = data["Close"].rolling(50).mean()
    delta = data["Close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = -delta.clip(upper=0).rolling(14).mean()
    data["RSI"] = 100 - (100 / (1 + gain / loss))

    ema12 = data["Close"].ewm(span=12).mean()
    ema26 = data["Close"].ewm(span=26).mean()
    data["MACD"] = ema12 - ema26
    data["MACD_Signal"] = data["MACD"].ewm(span=9).mean()

    data["BB_middle"] = data["Close"].rolling(20).mean()
    data["BB_upper"] = data["BB_middle"] + 2 * data["Close"].rolling(20).std()
    data["BB_lower"] = data["BB_middle"] - 2 * data["Close"].rolling(20).std()
    data["BB_position"] = (data["Close"] - data["BB_lower"]) / (data["BB_upper"] - data["BB_lower"])

    data["Return_1d"] = data["Close"].pct_change()
    data["Return_7d"] = data["Close"].pct_change(7)
    data["Return_14d"] = data["Close"].pct_change(14)
    data["Volatility"] = data["Return_1d"].rolling(14).std()

    # Strategjia
    data["Signal"] = 0
    data.loc[(data["SMA_20"] > data["SMA_50"]) & (data["RSI"] < 70), "Signal"] = 1
    data.loc[(data["SMA_20"] < data["SMA_50"]) | (data["RSI"] > 70), "Signal"] = -1

    # Backtest
    data["Returns"] = data["Close"].pct_change()
    data["Strategy"] = data["Signal"].shift(1) * data["Returns"]
    data["Cumulative_Market"] = (1 + data["Returns"]).cumprod()
    data["Cumulative_Strategy"] = (1 + data["Strategy"]).cumprod()

    return data.dropna()

@st.cache_data
def train_model(symbol, period):
    data = load_data(symbol, period)
    features = ["RSI", "MACD", "BB_position", "Return_1d", "Return_7d", "Return_14d", "Volatility"]
    data["Target"] = (data["Close"].shift(-1) > data["Close"]).astype(int)
    data = data.dropna()

    X = data[features]
    y = data["Target"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    model = RandomForestClassifier(n_estimators=200, max_depth=5, random_state=42)
    model.fit(X_train_s, y_train)
    acc = accuracy_score(y_test, model.predict(X_test_s))

    latest = scaler.transform(data[features].iloc[-1:])
    pred = model.predict(latest)[0]
    prob = model.predict_proba(latest)[0]

    return acc, pred, max(prob)

# Ngarko të dhënat
with st.spinner("Duke shkarkuar dhe analizuar..."):
    data = load_data(symbol, period)
    acc, pred, prob = train_model(symbol, period)

# Metrics kryesore
col1, col2, col3, col4 = st.columns(4)
col1.metric("💰 Çmimi aktual", f"${data['Close'].iloc[-1]:,.0f}")
col2.metric("📈 Buy & Hold", f"{data['Cumulative_Market'].iloc[-1]:.2f}x")
col3.metric("🤖 AI Bot", f"{data['Cumulative_Strategy'].iloc[-1]:.2f}x")
col4.metric("🎯 ML Accuracy", f"{acc*100:.1f}%")

# RSI dhe ML Sinjali
col5, col6 = st.columns(2)
rsi_val = data["RSI"].iloc[-1]
rsi_color = "🔴" if rsi_val > 70 else "🟢" if rsi_val < 30 else "🟡"
col5.markdown(f"### RSI: {rsi_color} {rsi_val:.1f}")
col6.markdown(f"### ML Sinjali: {'⬆️ RRITET' if pred == 1 else '⬇️ BIE'} ({prob*100:.1f}%)")

# Grafiku kryesor
st.subheader("📊 Çmimi dhe Indikatort")
fig, axes = plt.subplots(3, 1, figsize=(12, 10))

# Çmimi + SMA
axes[0].plot(data["Close"], label="Çmimi", color="blue", linewidth=1)
axes[0].plot(data["SMA_20"], label="SMA 20", color="orange", linewidth=1)
axes[0].plot(data["SMA_50"], label="SMA 50", color="red", linewidth=1)
axes[0].fill_between(data.index, data["BB_upper"], data["BB_lower"], alpha=0.1, color="gray", label="Bollinger Bands")
axes[0].set_title(f"{symbol} — Çmimi")
axes[0].legend(fontsize=8)
axes[0].grid(True, alpha=0.3)

# RSI
axes[1].plot(data["RSI"], color="purple", linewidth=1)
axes[1].axhline(70, color="red", linestyle="--", alpha=0.7, label="Mbiblerë (70)")
axes[1].axhline(30, color="green", linestyle="--", alpha=0.7, label="Mbishitur (30)")
axes[1].fill_between(data.index, data["RSI"], 70, where=(data["RSI"] > 70), alpha=0.3, color="red")
axes[1].fill_between(data.index, data["RSI"], 30, where=(data["RSI"] < 30), alpha=0.3, color="green")
axes[1].set_title("RSI")
axes[1].legend(fontsize=8)
axes[1].grid(True, alpha=0.3)

# Performanca
axes[2].plot(data["Cumulative_Market"], label="Buy & Hold", color="gray")
axes[2].plot(data["Cumulative_Strategy"], label="AI Bot", color="green")
axes[2].set_title("Performanca kumulative")
axes[2].legend(fontsize=8)
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
st.pyplot(fig)

# Sentiment Analysis
st.subheader("📰 Sentiment Analysis — Lajmet e Bitcoin")

@st.cache_data(ttl=3600)
def get_sentiment():
    API_KEY = "8b25ac4d45194b8ebb7611ec7f7df229"
    URL = f"https://newsapi.org/v2/everything?q=bitcoin&language=en&sortBy=publishedAt&pageSize=20&apiKey={API_KEY}"
    try:
        response = requests.get(URL, timeout=10)
        articles = response.json().get("articles", [])
    except:
        articles = []
    
    results = []
    for article in articles[:10]:
        title = article.get("title", "")
        date = article.get("publishedAt", "")[:10]
        polarity = TextBlob(title).sentiment.polarity
        if polarity > 0.1:
            emoji = "🟢"
            label = "Pozitiv"
        elif polarity < -0.1:
            emoji = "🔴"
            label = "Negativ"
        else:
            emoji = "🟡"
            label = "Neutral"
        results.append({
            "Sinjali": f"{emoji} {label}",
            "Titulli": title[:80],
            "Data": date,
            "Score": round(polarity, 3)
        })
    return results

# Shto imports në krye të dashboard.py
# import requests
# from textblob import TextBlob

with st.spinner("Duke shkarkuar lajmet..."):
    news = get_sentiment()

if news:
    df_news = pd.DataFrame(news)
    avg_sent = df_news["Score"].mean()
    
    col_s1, col_s2 = st.columns(2)
    if avg_sent > 0.1:
        col_s1.success(f"🟢 BULLISH — Sentiment: {avg_sent:+.3f}")
    elif avg_sent < -0.1:
        col_s1.error(f"🔴 BEARISH — Sentiment: {avg_sent:+.3f}")
    else:
        col_s1.warning(f"🟡 NEUTRAL — Sentiment: {avg_sent:+.3f}")
    
    col_s2.metric("📰 Lajme të analizuara", len(news))
    st.dataframe(df_news, use_container_width=True, hide_index=True)
    # Paper Trading Section
st.markdown("---")
st.subheader("💰 Paper Trading — Portofoli Virtual")

PORTFOLIO_FILE = "portfolio.json"

def load_portfolio():
    if os.path.exists(PORTFOLIO_FILE):
        with open(PORTFOLIO_FILE, "r") as f:
            return json.load(f)
    return {"cash": 10000.0, "holdings": {}, "trades": [], "initial_capital": 10000.0}

def get_current_prices(holdings):
    prices = {}
    for symbol in holdings:
        try:
            d = yf.download(symbol, period="1d", interval="1m", progress=False)
            prices[symbol] = float(d["Close"].iloc[-1].values[0])
        except:
            prices[symbol] = holdings[symbol]["avg_price"]
    return prices

portfolio = load_portfolio()
prices = get_current_prices(portfolio["holdings"])

# Llogarit vlerat
total_holdings = sum(
    portfolio["holdings"][s]["shares"] * prices.get(s, portfolio["holdings"][s]["avg_price"])
    for s in portfolio["holdings"]
)
total_value = portfolio["cash"] + total_holdings
total_profit = total_value - portfolio["initial_capital"]
pct = (total_profit / portfolio["initial_capital"]) * 100

# Metrics
col_p1, col_p2, col_p3, col_p4 = st.columns(4)
col_p1.metric("💵 Cash", f"${portfolio['cash']:,.2f}")
col_p2.metric("📦 Asetet", f"${total_holdings:,.2f}")
col_p3.metric("💼 Totali", f"${total_value:,.2f}")
col_p4.metric("📈 Fitim/Humbje", f"${total_profit:+.2f}", f"{pct:+.1f}%")

# Holdings tabela
if portfolio["holdings"]:
    st.markdown("**📦 Pozicionet aktuale:**")
    rows = []
    for symbol, data in portfolio["holdings"].items():
        current = prices.get(symbol, data["avg_price"])
        value = data["shares"] * current
        profit = (current - data["avg_price"]) * data["shares"]
        pct_h = ((current - data["avg_price"]) / data["avg_price"]) * 100
        rows.append({
            "Asset": symbol,
            "Shares": round(data["shares"], 6),
            "Çmimi mesatar": f"${data['avg_price']:,.2f}",
            "Çmimi aktual": f"${current:,.2f}",
            "Vlera": f"${value:,.2f}",
            "P&L": f"${profit:+.2f} ({pct_h:+.1f}%)"
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

# Historiku i tregtimeve
if portfolio["trades"]:
    st.markdown("**📋 Historiku i tregtimeve:**")
    trades_df = pd.DataFrame(portfolio["trades"][-10:])
    st.dataframe(trades_df, use_container_width=True, hide_index=True)
st.markdown("---")
st.caption("⚠️ Ky bot është për qëllime edukative dhe portofoli. Jo këshillë financiare.")