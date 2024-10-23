import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob

df = pd.read_csv("winemag-data.csv")

analyzer = SentimentIntensityAnalyzer()

df["vader sentiment"] = df['description'].apply(lambda x: analyzer.polarity_scores(x)['compound'])

df['blob sentiment'] = df['description'].apply(lambda x: TextBlob(x).sentiment.polarity)

df['vader sentiment'] = df['vader sentiment'].apply(lambda x: 'positive' if x > 0.05 
                                        else ('negative' if x < -0.05 
                                        else 'neutral'))

df['blob sentiment'] = df['blob sentiment'].apply(lambda x: 'positive' if x > 0 
                                        else ('negative' if x < 0 
                                        else 'neutral'))