import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

# Shkarko të dhënat
print("Duke shkarkuar të dhënat...")
data = yf.download("BTC-USD", period="1y", interval="1d")
data = pd.DataFrame(data["Close"]["BTC-USD"])
data.columns = ["Close"]

# Indikatort
data["SMA_20"] = data["Close"].rolling(20).mean()
data["SMA_50"] = data["Close"].rolling(50).mean()
delta = data["Close"].diff()
gain = delta.clip(lower=0).rolling(14).mean()
loss = -delta.clip(upper=0).rolling(14).mean()
data["RSI"] = 100 - (100 / (1 + gain / loss))

# Strategjia: Blej kur SMA_20 kalon SMA_50 dhe RSI < 70
# Shit kur SMA_20 bie nën SMA_50 ose RSI > 70
data["Signal"] = 0
data.loc[(data["SMA_20"] > data["SMA_50"]) & (data["RSI"] < 70), "Signal"] = 1   # Blej
data.loc[(data["SMA_20"] < data["SMA_50"]) | (data["RSI"] > 70), "Signal"] = -1  # Shit

# Backtest i thjeshtë
data["Returns"] = data["Close"].pct_change()
data["Strategy"] = data["Signal"].shift(1) * data["Returns"]
data["Cumulative_Market"] = (1 + data["Returns"]).cumprod()
data["Cumulative_Strategy"] = (1 + data["Strategy"]).cumprod()

# Rezultati
final_market = data["Cumulative_Market"].iloc[-1]
final_strategy = data["Cumulative_Strategy"].iloc[-1]
print(f"📈 Bitcoin (buy & hold): {final_market:.2f}x")
print(f"🤖 Boti ynë: {final_strategy:.2f}x")

# Grafiku
plt.figure(figsize=(12,6))
plt.plot(data["Cumulative_Market"], label="Buy & Hold", color="gray")
plt.plot(data["Cumulative_Strategy"], label="AI Bot", color="green")
plt.title("AI Trading Bot vs Buy & Hold")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("backtest.png")
print("✅ Grafiku u ruajt: backtest.png")