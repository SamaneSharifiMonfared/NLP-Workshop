import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('ggplot')
from nltk.corpus import stopwords
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import STOPWORDS
import spacy
from string import punctuation



df = pd.read_csv("tweet-sentiment-extraction/train.csv")



df['text'] = df['text'].str.lower()

df = df.dropna()

length = []
for x in df.text:
    length.append(len(x))

df['length'] = length

df_sorted = df.sort_values('length')

df = df_sorted

stopwords = list(STOPWORDS)
nlp = spacy.load('en_core_web_sm')

all_texts = " " 

for s in df['text']:
    all_texts = all_texts + " " + s


doc = nlp(all_texts[:1000000])

tokens = [token.text for token in doc]


punctuation  = punctuation + '\n'

word_frequencies = {}

for word in doc:
    if word.text not in stopwords:
        if word.text not in punctuation:
            if word.text not in word_frequencies.keys():
                word_frequencies[word.text] = 1
            else:
                word_frequencies[word.text] += 1

max_freq = max(word_frequencies.values())

# normalizing it
for word in word_frequencies.keys():
    word_frequencies[word] = word_frequencies[word]/max_freq


sentece_tokens = [sent for sent in doc.sents]


sentence_score = {}

for sent in sentece_tokens:
    for word in sent:
        if word.text in word_frequencies.keys():
            if sent not in sentence_score.keys():
                sentence_score[sent] = word_frequencies[word.text]
            else:
                sentence_score[sent] += word_frequencies[word.text]


print(sentence_score)

print(df.columns)

column_names = ['Sentence', 'Score']

df_score =  pd.DataFrame(sentence_score,  index=[0,1]).T

print(df_score.columns)

print(df_score)

df_score.columns = column_names

df_score=df_score[1400:]



df_score = df_score.sort_values('Score')


df_score.to_csv("output/text_scored.csv")

