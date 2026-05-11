import requests
from textblob import TextBlob
import pandas as pd

API_KEY = "8b25ac4d45194b8ebb7611ec7f7df229"
URL = f"https://newsapi.org/v2/everything?q=bitcoin&language=en&sortBy=publishedAt&pageSize=20&apiKey={API_KEY}"

print("Duke shkarkuar lajmet e Bitcoin...")

try:
    response = requests.get(URL, timeout=10)
    data = response.json()
    articles = data.get("articles", [])
except Exception as e:
    print(f"Gabim: {e}")
    articles = []

if not articles:
    print("Nuk u gjetën lajme!")
else:
    print(f"\n📰 {len(articles)} lajme të gjetura\n")
    print("-" * 60)

    sentiments = []
    for article in articles[:10]:
        title = article.get("title", "")
        date = article.get("publishedAt", "")[:10]

        blob = TextBlob(title)
        polarity = blob.sentiment.polarity

        if polarity > 0.1:
            emoji = "🟢 POZITIV"
        elif polarity < -0.1:
            emoji = "🔴 NEGATIV"
        else:
            emoji = "🟡 NEUTRAL"

        sentiments.append(polarity)
        print(f"{emoji} ({polarity:+.2f})")
        print(f"📰 {title[:70]}")