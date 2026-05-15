import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

ASSETS = {
    "BTC-USD": "Bitcoin",
    "ETH-USD": "Ethereum", 
    "AAPL": "Apple",
    "TSLA": "Tesla",
    "NVDA": "NVIDIA",
    "GLD": "Gold ETF"
}

def get_stats(symbol, period="1y"):
    data = yf.download(symbol, period=period, interval="1d", progress=False)
    data = pd.DataFrame(data["Close"][symbol])
    data.columns = ["Close"]

    data["SMA_20"] = data["Close"].rolling(20).mean()
    data["SMA_50"] = data["Close"].rolling(50).mean()
    delta = data["Close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = -delta.clip(upper=0).rolling(14).mean()
    data["RSI"] = 100 - (100 / (1 + gain / loss))
    data = data.dropna()

    data["Signal"] = 0
    data.loc[(data["SMA_20"] > data["SMA_50"]) & (data["RSI"] < 70), "Signal"] = 1
    data.loc[(data["SMA_20"] < data["SMA_50"]) | (data["RSI"] > 70), "Signal"] = -1

    data["Returns"] = data["Close"].pct_change()
    data["Strategy_Returns"] = data["Signal"].shift(1) * data["Returns"]
    data = data.dropna()

    # Stats
    sharpe = np.sqrt(252) * data["Strategy_Returns"].mean() / data["Strategy_Returns"].std()
    cumulative = (1 + data["Strategy_Returns"]).cumprod()
    market_cum = (1 + data["Returns"]).cumprod()
    rolling_max = cumulative.expanding().max()
    max_dd = ((cumulative - rolling_max) / rolling_max).min()
    win_rate = (data["Strategy_Returns"] > 0).sum() / (data["Strategy_Returns"] != 0).sum()
    total_ret = cumulative.iloc[-1] - 1
    market_ret = market_cum.iloc[-1] - 1
    current_price = data["Close"].iloc[-1]
    rsi_now = data["RSI"].iloc[-1]

    if data["Signal"].iloc[-1] == 1:
        signal = "🟢 BUY"
    elif data["Signal"].iloc[-1] == -1:
        signal = "🔴 SELL"
    else:
        signal = "🟡 HOLD"

    return {
        "symbol": symbol,
        "name": ASSETS.get(symbol, symbol),
        "price": current_price,
        "rsi": rsi_now,
        "signal": signal,
        "bot_return": total_ret,
        "market_return": market_ret,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "win_rate": win_rate,
        "cumulative": cumulative,
        "market_cumulative": market_cum
    }

print("🌐 MULTI-ASSET ANALYSIS\n")
print("Duke analizuar të gjitha asetet...")

results = []
cumulative_data = {}

for symbol in ASSETS:
    try:
        stats = get_stats(symbol)
        results.append(stats)
        cumulative_data[symbol] = stats["cumulative"]
        print(f"✅ {symbol} u analizua")
    except Exception as e:
        print(f"❌ {symbol} gabim: {e}")

# Tabela krahasuese
print("\n" + "="*80)
print(f"{'Asset':<8} {'Çmimi':>10} {'RSI':>6} {'Sinjali':>12} {'Bot%':>8} {'Mkt%':>8} {'Sharpe':>8} {'WinRate':>8}")
print("="*80)

for r in results:
    print(f"{r['symbol']:<8} ${r['price']:>9,.2f} {r['rsi']:>6.1f} {r['signal']:>12} "
          f"{r['bot_return']*100:>+7.1f}% {r['market_return']*100:>+7.1f}% "
          f"{r['sharpe']:>8.2f} {r['win_rate']*100:>7.1f}%")

# Ranko asetet sipas performancës
print("\n🏆 RANKING — Asetet më të mira (Bot Return):")
sorted_results = sorted(results, key=lambda x: x["bot_return"], reverse=True)
for i, r in enumerate(sorted_results, 1):
    emoji = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"#{i}"
    print(f"  {emoji} {r['name']}: {r['bot_return']*100:+.1f}%")

# Grafiku krahasues
fig, axes = plt.subplots(2, 1, figsize=(14, 10))

# Performanca kumulative
for r in results:
    axes[0].plot(r["cumulative"], label=f"{r['symbol']} Bot", linewidth=1.5)
axes[0].axhline(1, color="black", linestyle="--", alpha=0.3)
axes[0].set_title("🤖 AI Bot — Performanca Kumulative e të Gjitha Aseteve")
axes[0].legend(fontsize=8)
axes[0].grid(True, alpha=0.3)

# RSI krahasues
rsi_vals = [r["rsi"] for r in results]
symbols = [r["symbol"] for r in results]
colors = ["red" if v > 70 else "green" if v < 30 else "orange" for v in rsi_vals]
bars = axes[1].bar(symbols, rsi_vals, color=colors, alpha=0.7)
axes[1].axhline(70, color="red", linestyle="--", alpha=0.7, label="Mbiblerë (70)")
axes[1].axhline(30, color="green", linestyle="--", alpha=0.7, label="Mbishitur (30)")