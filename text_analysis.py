from textblob import TextBlob

def analyze_sentiment(text):
    """Analyze sentiment from text using TextBlob"""
    blob = TextBlob(text)
    sentiment = blob.sentiment.polarity
    if sentiment < -0.5:
        return "negative"
    elif sentiment > 0.5:
        return "positive"
    else:
        return "neutral"