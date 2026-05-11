import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

print("Duke shkarkuar të dhënat...")
data = yf.download("BTC-USD", period="5y", interval="1d")
data = pd.DataFrame(data["Close"]["BTC-USD"])
data.columns = ["Close"]

# Features të zgjeruara
data["SMA_20"] = data["Close"].rolling(20).mean()
data["SMA_50"] = data["Close"].rolling(50).mean()
data["SMA_200"] = data["Close"].rolling(200).mean()

# RSI
delta = data["Close"].diff()
gain = delta.clip(lower=0).rolling(14).mean()
loss = -delta.clip(upper=0).rolling(14).mean()
data["RSI"] = 100 - (100 / (1 + gain / loss))

# MACD
ema12 = data["Close"].ewm(span=12).mean()
ema26 = data["Close"].ewm(span=26).mean()
data["MACD"] = ema12 - ema26
data["MACD_Signal"] = data["MACD"].ewm(span=9).mean()

# Bollinger Bands
data["BB_middle"] = data["Close"].rolling(20).mean()
data["BB_upper"] = data["BB_middle"] + 2 * data["Close"].rolling(20).std()
data["BB_lower"] = data["BB_middle"] - 2 * data["Close"].rolling(20).std()
data["BB_position"] = (data["Close"] - data["BB_lower"]) / (data["BB_upper"] - data["BB_lower"])

# Returns
data["Return_1d"] = data["Close"].pct_change()
data["Return_3d"] = data["Close"].pct_change(3)
data["Return_7d"] = data["Close"].pct_change(7)
data["Return_14d"] = data["Close"].pct_change(14)

# Volatility
data["Volatility_7"] = data["Return_1d"].rolling(7).std()
data["Volatility_14"] = data["Return_1d"].rolling(14).std()

# Target
data["Target"] = (data["Close"].shift(-1) > data["Close"]).astype(int)
data = data.dropna()

# Features
features = [
    "SMA_20", "SMA_50", "SMA_200", "RSI",
    "MACD", "MACD_Signal", "BB_position",
    "Return_1d", "Return_3d", "Return_7d", "Return_14d",
    "Volatility_7", "Volatility_14"
]

X = data[features]
y = data["Target"]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

# Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Dy modele - zgjedhim më të mirën
print("Duke trajnuar modelet...")

rf = RandomForestClassifier(n_estimators=200, max_depth=5, random_state=42)
rf.fit(X_train_scaled, y_train)
rf_acc = accuracy_score(y_test, rf.predict(X_test_scaled))

gb = GradientBoostingClassifier(n_estimators=200, max_depth=3, random_state=42)
gb.fit(X_train_scaled, y_train)
gb_acc = accuracy_score(y_test, gb.predict(X_test_scaled))

print(f"\n🌲 Random Forest:        {rf_acc*100:.1f}%")
print(f"🚀 Gradient Boosting:    {gb_acc*100:.1f}%")

# Përdor modelin më të mirë
best_model = rf if rf_acc > gb_acc else gb
best_name = "Random Forest" if rf_acc > gb_acc else "Gradient Boosting"
best_acc = max(rf_acc, gb_acc)
print(f"\n✅ Modeli më i mirë: {best_name} ({best_acc*100:.1f}%)")

# Parashikimi për nesër
latest_scaled = scaler.transform(data[features].iloc[-1:])
prediction = best_model.predict(latest_scaled)[0]
probability = best_model.predict_proba(latest_scaled)[0]

print(f"\n🔮 Parashikimi për nesër:")
print(f"{'⬆️  RRITET' if prediction == 1 else '⬇️  BIE'}")
print(f"Probabiliteti: {max(probability)*100:.1f}%")

# Feature importance
print(f"\n🔍 Faktorët më të rëndësishëm:")
importance = pd.DataFrame({
    "Feature": features,
    "Importance": best_model.feature_importances_
}).sort_values("Importance", ascending=False).head(5)
print(importance.to_string(index=False))