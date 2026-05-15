import yfinance as yf
import pandas as pd
import numpy as np

def calculate_stats(symbol="BTC-USD", period="1y"):
    print(f"Duke llogaritur statistikat për {symbol}...\n")
    
    # Shkarko të dhënat
    data = yf.download(symbol, period=period, interval="1d", progress=False)
    data = pd.DataFrame(data["Close"][symbol])
    data.columns = ["Close"]

    # Indikatorët
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

    # Returns
    data = data.dropna()
    data["Returns"] = data["Close"].pct_change()
    data["Strategy_Returns"] = data["Signal"].shift(1) * data["Returns"]

    # ============================================
    # SHARPE RATIO
    # ============================================
    risk_free_rate = 0.05 / 252  # 5% vjetor i konvertuar në ditor
    excess_returns = data["Strategy_Returns"] - risk_free_rate
    sharpe = np.sqrt(252) * excess_returns.mean() / excess_returns.std()

    # ============================================
    # MAX DRAWDOWN
    # ============================================
    cumulative = (1 + data["Strategy_Returns"]).cumprod()
    rolling_max = cumulative.expanding().max()
    drawdown = (cumulative - rolling_max) / rolling_max
    max_drawdown = drawdown.min()

    # ============================================
    # WIN RATE
    # ============================================
    trades = data[data["Signal"].diff() != 0].copy()
    winning_trades = (data["Strategy_Returns"] > 0).sum()
    total_trades = (data["Strategy_Returns"] != 0).sum()
    win_rate = winning_trades / total_trades if total_trades > 0 else 0

    # ============================================
    # VOLATILITY
    # ============================================
    strategy_vol = data["Strategy_Returns"].std() * np.sqrt(252)
    market_vol = data["Returns"].std() * np.sqrt(252)

    # ============================================
    # TOTAL RETURN
    # ============================================
    total_return_strategy = (1 + data["Strategy_Returns"]).cumprod().iloc[-1] - 1
    total_return_market = (1 + data["Returns"]).cumprod().iloc[-1] - 1

    # ============================================
    # BEST / WORST DAY
    # ============================================
    best_day = data["Strategy_Returns"].max()
    worst_day = data["Strategy_Returns"].min()

    # ============================================
    # CALMAR RATIO (Return / Max Drawdown)
    # ============================================
    calmar = total_return_strategy / abs(max_drawdown) if max_drawdown != 0 else 0

    # ============================================
    # REZULTATET
    # ============================================
    print("=" * 50)
    print(f"📊 STATISTIKAT E AVANCUARA — {symbol}")
    print("=" * 50)

    print(f"\n📈 KTHIMET:")
    print(f"  🤖 AI Bot:        {total_return_strategy*100:>+8.1f}%")
    print(f"  📊 Buy & Hold:    {total_return_market*100:>+8.1f}%")
    print(f"  💡 Avantazhi:     {(total_return_strategy-total_return_market)*100:>+8.1f}%")

    print(f"\n⚡ RREZIKU:")
    sharpe_emoji = "🟢" if sharpe > 1 else "🟡" if sharpe > 0 else "🔴"
    print(f"  {sharpe_emoji} Sharpe Ratio:   {sharpe:>8.2f}  (>1.0 = i mirë)")
    
    drawdown_emoji = "🟢" if max_drawdown > -0.1 else "🟡" if max_drawdown > -0.2 else "🔴"
    print(f"  {drawdown_emoji} Max Drawdown:   {max_drawdown*100:>8.1f}%  (sa keq shkoi)")
    
    print(f"  📉 Volatility Bot: {strategy_vol*100:>7.1f}%")
    print(f"  📉 Volatility Mkt: {market_vol*100:>7.1f}%")

    print(f"\n🎯 TREGTIMI:")
    winrate_emoji = "🟢" if win_rate > 0.55 else "🟡" if win_rate > 0.45 else "🔴"
    print(f"  {winrate_emoji} Win Rate:       {win_rate*100:>8.1f}%  (>55% = i mirë)")
    print(f"  📋 Total trades:  {total_trades:>8}")
    print(f"  📈 Dita më e mirë: {best_day*100:>+7.1f}%")
    print(f"  📉 Dita më e keqe: {worst_day*100:>+7.1f}%")

    print(f"\n🏆 CALMAR RATIO:    {calmar:>8.2f}  (>1.0 = excellent)")

    # Vlerësimi i përgjithshëm
    score = 0
    if sharpe > 1: score += 1
    if max_drawdown > -0.15: score += 1
    if win_rate > 0.52: score += 1
    if total_return_strategy > total_return_market: score += 1
    if calmar > 1: score += 1

    print(f"\n{'='*50}")
    ratings = {5: "🏆 EXCELLENT", 4: "🟢 SHUMË MIRË", 3: "🟡 MIRË", 2: "🟠 MESATAR", 1: "🔴 DOBËT"}
    print(f"⭐ VLERËSIMI: {ratings.get(score, '🔴 DOBËT')} ({score}/5)")
    print(f"{'='*50}")

    return {
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
        "win_rate": win_rate,
        "total_return": total_return_strategy,
        "market_return": total_return_market,
        "volatility": strategy_vol,
        "calmar": calmar,
        "score": score
    }

if __name__ == "__main__":
    # Testo për BTC dhe ETH
    for symbol in ["BTC-USD", "ETH-USD"]:
        calculate_stats(symbol, "1y")
        print()