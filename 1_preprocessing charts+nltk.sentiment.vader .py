import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('ggplot')

import nltk

# Read first texts

df = pd.read_csv("tweet-sentiment-extraction/train.csv")

print(df.columns)

df = df.dropna(inplace = True)


df['text'] = df['text'].str.lower()
df['selected_text'] = df['selected_text'].str.lower()
df['sentiment'] = df['selected_text'].str.lower()

df_most_freq = df['text'].value_counts().head(20)

ax = df_most_freq.plot(kind = "pie", title="text",
                                              figsize=(10 , 5))
plt.show()




df_most_freq_2 = df['text'].value_counts().sort_values().head(20)

ax = df_most_freq_2.plot(kind = "pie", title="Most frequent texts after preprocessing",
                                              figsize=(10 , 10))

plt.show()


print(df)

print(df_most_freq_2)



df_word_tkn = nltk.word_tokenize(df['text'][0])

df_word_tag = nltk.pos_tag(df_word_tkn)

ent = nltk.chunk.ne_chunk(df_word_tag)


from tqdm.notebook import tqdm
from nltk.sentiment.vader import SentimentIntensityAnalyzer as SIA

sia = SIA()

res = {}

for i ,row in enumerate(tqdm(df.text , total = len(df))):
    score = sia.polarity_scores(row)
    res[row] = score['compound']

score_df = pd.DataFrame(res ,  index=[0]).T
score_df = score_df.set_axis(['score'], axis=1)

score_df.to_csv("output/sentiment_scores_vader.csv")
