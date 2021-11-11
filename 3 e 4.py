
largestMemb = max(df.animeNumMembers) #df.animeNumMembers.iloc[df.animeNumMembers.idxmax(axis=1)]
largestUsers =  max(df.animeUsers.apply(lambda x: pd.to_numeric(x, errors='coerce'))) #int(df.animeUsers.iloc[df.animeUsers.apply(lambda x: pd.to_numeric(x, errors='coerce')).idxmax(axis=1)])
highestVote =  max(df.animeScore.apply(lambda x: pd.to_numeric(x, errors='coerce')))


df["normMemb"] = np.array(df.animeNumMembers) / largestMemb
df["normUsers"] = np.array(df.animeUsers.apply(lambda x: pd.to_numeric(x, errors='coerce'))) / largestUsers
df["normVote"] = np.array(df.animeScore.apply(lambda x: pd.to_numeric(x, errors='coerce'))) / highestVote

cap = lambda x: 1.0 if x>1 else x
cosine = 1231231 # 0.80
score = cap(4/5 * (cosine) + 1/10 * (df.normMemb[0] + df.normUsers[0] + df.normVote[0]))

query = "death note"


from nltk.sentiment import SentimentIntensityAnalyzer
import operator
sia = SentimentIntensityAnalyzer()
df["sentiment_score"] = df.animeDescription.apply(lambda x: sia.polarity_scores(x)["compound"])

df["sentiment"] = np.select([df["sentiment_score"] < 0, df["sentiment_score"] == 0, df["sentiment_score"] > 0],
                           ['neg', 'neu', 'pos'])
