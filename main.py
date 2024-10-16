import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import string


df = pd.read_csv('tweet-sentiment-extraction/train.csv')

# print(df.to_string()) 
# print(df.columns)
# print(df.shape)
# print(df.head())
# # Sort the data based on column value sort_values(“name_of_column”)
# print(df.sort_values("textID"))

print(df.describe())
print(df.head)


# Handling Missing Values

print(df.isnull().sum())

# Dropping is best option here as there is only on missing value

df.dropna(inplace = True)


#  Distribution of Sentiments

print(df['sentiment'].value_counts())

# show a plot
plt.figure(figsize = (13,5))
plt.subplot(121)
plt.title('Distribution of sentiments in df Data')
sns.countplot(df['sentiment'])
plt.show()

# Cleaning the Data

def clean_data(data):
    # Removing extra spaces in the beginning of text
    data = data.strip()
    # Lower the Text
    data = data.lower()
    return data

df['text'] = df['text'].apply(lambda x: clean_data(x))


# Ngram Analysis
# It is most easiest and beautiful way to analyze most frequent words occuring in our Dataset

# Ngram Analysis is done using CountVectorizer of Sklearn Library and its documentation can be seen and understood here: https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html

positive_tweet = df[df['sentiment']=='positive']
negative_tweet = df[df['sentiment']=='negative']
neutral_tweet = df[df['sentiment']=='neutral']

print(positive_tweet)

from sklearn.feature_extraction.text import CountVectorizer

def get_top_n_words(corpus, ngram_range = (1,1), n = None):
    vec = CountVectorizer(ngram_range = ngram_range, stop_words = 'english').fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis = 0)
    word_freq = [(word, sum_words[0,idx]) for word, idx in vec.vocabulary_.items()]
    word_freq = sorted(word_freq, key = lambda x: x[1], reverse = True)
    return word_freq[:n]


# Unigram Analysis of text column

pos_unigram = get_top_n_words(positive_tweet['text'], (1,1), 20)
neutral_unigram = get_top_n_words(neutral_tweet['text'], (1,1), 20)
neg_unigram = get_top_n_words(negative_tweet['text'], (1,1), 20)

df1 = pd.DataFrame(pos_unigram, columns = ['word','count'])
df2 = pd.DataFrame(neutral_unigram, columns = ['word','count'])
df3 = pd.DataFrame(neg_unigram, columns = ['word','count'])

plt.tight_layout()
fig,(ax1,ax2,ax3)=plt.subplots(3,1,figsize=(12,17))
sns.barplot(x = 'count' , y = 'word', data = df1, orient = 'h',ax = ax1)
ax1.set_title('Most repititve words in positive tweets')
ax1.spines["right"].set_visible(False)
ax1.spines["top"].set_visible(False)
ax1.grid(False)
sns.barplot(x = 'count' , y = 'word', data = df2, orient = 'h',ax = ax2)
ax2.set_title('Most repititve words in neutral tweets')
ax2.spines["right"].set_visible(False)
ax2.spines["top"].set_visible(False)
ax2.grid(False)
sns.barplot(x = 'count' , y = 'word', data = df3, orient = 'h',ax = ax3)
ax3.set_title('Most repititve words in negative tweets')
ax3.spines["right"].set_visible(False)
ax3.spines["top"].set_visible(False)
ax3.grid(False)
plt.show()



