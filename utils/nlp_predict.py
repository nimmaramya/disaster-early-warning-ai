import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

# download once (safe for deployment)
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except:
    nltk.download('vader_lexicon')

sia = SentimentIntensityAnalyzer()


# -------------------------------------------------
# DISASTER KEYWORDS + WEIGHTS
# -------------------------------------------------

DISASTER_KEYWORDS = {
    "flood": 0.4,
    "flooding": 0.4,
    "heavy rain": 0.35,
    "heavy rainfall": 0.4,
    "cyclone": 0.5,
    "storm": 0.2,
    "overflow": 0.3,
    "waterlogging": 0.15,
    "dam break": 0.3,
    "landslide": 0.25,
    "evacuation": 0.2,
    "rescue": 0.15,
    "disaster": 0.25,
    "emergency": 0.4
}


# -------------------------------------------------
# TEXT CLEANING
# -------------------------------------------------

def preprocess(text):
    return text.lower().strip()


# -------------------------------------------------
# KEYWORD BOOST
# -------------------------------------------------

def keyword_boost(text):
    boost = 0

    for word, weight in DISASTER_KEYWORDS.items():
        if word in text:
            boost += weight

    return min(boost, 0.6)  # cap boost


# -------------------------------------------------
# MAIN NLP RISK FUNCTION
# -------------------------------------------------

def predict_text_risk(text):

    text = preprocess(text)

    # sentiment analysis
    sentiment = sia.polarity_scores(text)

    negative = sentiment['neg']
    compound = sentiment['compound']

    # convert sentiment → disaster signal
    # (compound ranges -1 to +1)
    sentiment_score = max(0, -compound)  # only negative matters

    # keyword boost
    boost = keyword_boost(text)

    # combine
    risk_score = sentiment_score * 0.7 + boost

    return float(min(1.0, risk_score))