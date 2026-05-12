import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import json
import os

# ============================================
# PAPER TRADING SIMULATOR
# ============================================

PORTFOLIO_FILE = "portfolio.json"

def load_portfolio():
    """Ngarko portfolion nga skedari"""
    if os.path.exists(PORTFOLIO_FILE):
        with open(PORTFOLIO_FILE, "r") as f:
            return json.load(f)
    return {
        "cash": 10000.0,        # Fillojmë me $10,000 virtuale
        "holdings": {},          # Asetet që mbajmë
        "trades": [],            # Historiku i tregtimeve
        "initial_capital": 10000.0
    }

def save_portfolio(portfolio):
    """Ruaj portfolion në skedar"""
    with open(PORTFOLIO_FILE, "w") as f:
        json.dump(portfolio, f, indent=2)

def get_signal(symbol):
    """Merr sinjalin e blerjes/shitjes"""
    data = yf.download(symbol, period="3mo", interval="1d", progress=False)
    data = pd.DataFrame(data["Close"][symbol])
    data.columns = ["Close"]

    # Indikatorët
    data["SMA_20"] = data["Close"].rolling(20).mean()
    data["SMA_50"] = data["Close"].rolling(50).mean()
    delta = data["Close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = -delta.clip(upper=0).rolling(14).mean()
    data["RSI"] = 100 - (100 / (1 + gain / loss))
    data = data.dropna()

    latest = data.iloc[-1]
    price = latest["Close"]

    if latest["SMA_20"] > latest["SMA_50"] and latest["RSI"] < 70:
        signal = "BUY"
    elif latest["SMA_20"] < latest["SMA_50"] or latest["RSI"] > 70:
        signal = "SELL"
    else:
        signal = "HOLD"

    return signal, price, latest["RSI"]

def buy(portfolio, symbol, amount_usd):
    """Bli asset"""
    signal, price, rsi = get_signal(symbol)
    
    if portfolio["cash"] < amount_usd:
        print(f"❌ Nuk ke mjaft cash! Ke ${portfolio['cash']:.2f}")
        return portfolio
    
    shares = amount_usd / price
    portfolio["cash"] -= amount_usd
    
    if symbol in portfolio["holdings"]:
        old = portfolio["holdings"][symbol]
        total_shares = old["shares"] + shares
        avg_price = (old["shares"] * old["avg_price"] + shares * price) / total_shares
        portfolio["holdings"][symbol] = {"shares": total_shares, "avg_price": avg_price}
    else:
        portfolio["holdings"][symbol] = {"shares": shares, "avg_price": price}
    
    trade = {
        "type": "BUY",
        "symbol": symbol,
        "shares": round(shares, 6),
        "price": round(price, 2),
        "total": round(amount_usd, 2),
        "date": datetime.now().strftime("%Y-%m-%d %H:%M")
    }
    portfolio["trades"].append(trade)
    
    print(f"✅ BLERJE: {shares:.6f} {symbol} @ ${price:,.2f}")
    print(f"💵 U shpenzuan: ${amount_usd:.2f}")
    print(f"💰 Cash mbetur: ${portfolio['cash']:.2f}")
    return portfolio

def sell(portfolio, symbol, percent=100):
    """Shit asset (percent = sa % të shesim)"""
    if symbol not in portfolio["holdings"]:
        print(f"❌ Nuk ke {symbol} në portofol!")
        return portfolio
    
    _, price, rsi = get_signal(symbol)
    holding = portfolio["holdings"][symbol]
    shares_to_sell = holding["shares"] * (percent / 100)
    total_value = shares_to_sell * price
    profit = (price - holding["avg_price"]) * shares_to_sell
    
    portfolio["cash"] += total_value
    
    if percent == 100:
        del portfolio["holdings"][symbol]
    else:
        portfolio["holdings"][symbol]["shares"] -= shares_to_sell
    
    trade = {
        "type": "SELL",
        "symbol": symbol,
        "shares": round(shares_to_sell, 6),
        "price": round(price, 2),
        "total": round(total_value, 2),
        "profit": round(profit, 2),
        "date": datetime.now().strftime("%Y-%m-%d %H:%M")
    }
    portfolio["trades"].append(trade)
    
    emoji = "📈" if profit > 0 else "📉"
    print(f"✅ SHITJE: {shares_to_sell:.6f} {symbol} @ ${price:,.2f}")
    print(f"💵 U morën: ${total_value:.2f}")
    print(f"{emoji} Fitim/Humbje: ${profit:+.2f}")
    print(f"💰 Cash total: ${portfolio['cash']:.2f}")
    return portfolio

def show_portfolio(portfolio):
    """Shfaq gjendjen e portfolios"""
    print("\n" + "="*50)
    print("📊 PORTOFOLI YT VIRTUAL")
    print("="*50)
    print(f"💵 Cash:           ${portfolio['cash']:>10,.2f}")
    
    total_holdings_value = 0
    if portfolio["holdings"]:
        print("\n📦 Asetet:")
        for symbol, data in portfolio["holdings"].items():
            _, current_price, _ = get_signal(symbol)
            value = data["shares"] * current_price
            profit = (current_price - data["avg_price"]) * data["shares"]
            total_holdings_value += value
            emoji = "📈" if profit > 0 else "📉"
            print(f"  {emoji} {symbol}: {data['shares']:.6f} shares")
            print(f"      Çmimi mesatar: ${data['avg_price']:,.2f}")
            print(f"      Çmimi aktual:  ${current_price:,.2f}")
            print(f"      Vlera:         ${value:,.2f} ({profit:+.2f})")
    
    total_value = portfolio["cash"] + total_holdings_value
    total_profit = total_value - portfolio["initial_capital"]
    pct = (total_profit / portfolio["initial_capital"]) * 100
    
    print(f"\n{'='*50}")
    print(f"💼 Vlera totale:   ${total_value:>10,.2f}")
    print(f"🎯 Kapitali fillestar: ${portfolio['initial_capital']:>10,.2f}")
    emoji = "📈" if total_profit > 0 else "📉"
    print(f"{emoji} Fitim/Humbje:  ${total_profit:>+10.2f} ({pct:+.1f}%)")
    
    if portfolio["trades"]:
        print(f"\n📋 Tregtimi i fundit:")
        last = portfolio["trades"][-1]
        print(f"  {last['type']} {last['symbol']} @ ${last['price']:,.2f} — {last['date']}")
    print("="*50)

def auto_trade(portfolio, symbols=["BTC-USD", "ETH-USD"]):
    """Boti tregton automatikisht bazuar në sinjale"""
    print("\n🤖 AUTO-TRADE — Duke analizuar tregun...\n")
    
    for symbol in symbols:
        signal, price, rsi = get_signal(symbol)
        print(f"📊 {symbol}: ${price:,.2f} | RSI: {rsi:.1f} | Sinjali: {signal}")
        
        if signal == "BUY" and portfolio["cash"] > 500:
            invest = min(portfolio["cash"] * 0.3, 2000)
            print(f"🟢 Blerje automatike: ${invest:.2f}")
            portfolio = buy(portfolio, symbol, invest)
        
        elif signal == "SELL" and symbol in portfolio["holdings"]:
            print(f"🔴 Shitje automatike: 100%")
            portfolio = sell(portfolio, symbol, 100)
        
        else:
            print(f"🟡 HOLD — Nuk ka veprim")
        print()
    
    return portfolio

# ============================================
# MAIN — Ekzekuto
# ============================================
if __name__ == "__main__":
    portfolio = load_portfolio()
    
    print("🤖 AI PAPER TRADING BOT")
    print("========================\n")
    
    # Auto-trade
    portfolio = auto_trade(portfolio, ["BTC-USD", "ETH-USD"])
    
    # Shfaq portfolion
    show_portfolio(portfolio)
    
    # Ruaj
    save_portfolio(portfolio)
    print("\n✅ Portofoli u ruajt në portfolio.json")