import feedparser


# disaster keywords
DISASTER_KEYWORDS = [
    "flood",
    "storm",
    "cyclone",
    "heavy rain",
    "disaster",
    "evacuation",
    "damage",
    "alert"
]


# -----------------------------
# FETCH GOOGLE NEWS
# -----------------------------

import feedparser
import urllib.parse

def fetch_news(city):

    query = f"{city} disaster flood storm cyclone"

    encoded_query = urllib.parse.quote(query)

    url = f"https://news.google.com/rss/search?q={encoded_query}"

    feed = feedparser.parse(url)

    headlines = []

    for entry in feed.entries[:5]:
        headlines.append(entry.title)

    return headlines

# -----------------------------
# KEYWORD INTENSITY
# -----------------------------

def disaster_keyword_score(text):

    score = 0

    text = text.lower()

    for word in DISASTER_KEYWORDS:
        if word in text:
            score += 2

    return score


# -----------------------------
# BUILD SOCIAL TEXT
# -----------------------------cc

def build_social_text(city):

    headlines = fetch_news(city)

    combined_text = " ".join(headlines)

    keyword_score = disaster_keyword_score(combined_text)

    return combined_text, keyword_score