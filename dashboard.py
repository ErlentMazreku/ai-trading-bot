import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="AI Trading Bot", page_icon="🤖", layout="wide")

st.title("🤖 AI Trading Bot Dashboard")
st.markdown("**Strategji bazuar në SMA + RSI**")

# Sidebar - kontrollet
st.sidebar.header("⚙️ Cilësimet")
symbol = st.sidebar.selectbox("Zgjidh assetin", ["BTC-USD", "ETH-USD", "AAPL", "TSLA"])
period = st.sidebar.selectbox("Periudha", ["6mo", "1y", "2y"])

# Shkarko të dhënat
with st.spinner("Duke shkarkuar të dhënat..."):
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

# Strategjia
data["Signal"] = 0
data.loc[(data["SMA_20"] > data["SMA_50"]) & (data["RSI"] < 70), "Signal"] = 1
data.loc[(data["SMA_20"] < data["SMA_50"]) | (data["RSI"] > 70), "Signal"] = -1

# Backtest
data["Returns"] = data["Close"].pct_change()
data["Strategy"] = data["Signal"].shift(1) * data["Returns"]
data["Cumulative_Market"] = (1 + data["Returns"]).cumprod()
data["Cumulative_Strategy"] = (1 + data["Strategy"]).cumprod()

final_market = data["Cumulative_Market"].iloc[-1]
final_strategy = data["Cumulative_Strategy"].iloc[-1]

# Metrics
col1, col2, col3 = st.columns(3)
col1.metric("💰 Çmimi aktual", f"${data['Close'].iloc[-1]:,.0f}")
col2.metric("📈 Buy & Hold", f"{final_market:.2f}x", f"{(final_market-1)*100:.1f}%")
col3.metric("🤖 AI Bot", f"{final_strategy:.2f}x", f"{(final_strategy-1)*100:.1f}%")

# RSI gauge
rsi_val = data["RSI"].iloc[-1]
rsi_color = "🔴" if rsi_val > 70 else "🟢" if rsi_val < 30 else "🟡"
st.markdown(f"### RSI aktual: {rsi_color} {rsi_val:.1f}")

# Grafiku kryesor
st.subheader("📊 Performanca: AI Bot vs Buy & Hold")
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

ax1.plot(data["Close"], label="Çmimi", color="blue", linewidth=1)
ax1.plot(data["SMA_20"], label="SMA 20", color="orange", linewidth=1)
ax1.plot(data["SMA_50"], label="SMA 50", color="red", linewidth=1)
ax1.set_title(f"{symbol} — Çmimi dhe SMA")
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.plot(data["Cumulative_Market"], label="Buy & Hold", color="gray")
ax2.plot(data["Cumulative_Strategy"], label="AI Bot", color="green")
ax2.set_title("Performanca kumulative")
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
st.pyplot(fig)

st.markdown("---")
st.caption("⚠️ Ky bot është për qëllime edukative dhe portofoli. Jo këshillë financiare.")