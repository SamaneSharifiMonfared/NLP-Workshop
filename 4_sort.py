import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from scipy.special import softmax
plt.style.use('ggplot')
import nltk


df_new = pd.read_csv("scores.csv")
df = pd.DataFrame(df_new)
# print(df)

length = []

for x in df.text:
    length.append(len(x))

print(max(length))
df['length'] = length
print(df)


df_sorted = df.sort_values('length')

df_sorted.to_csv("output/sorted.csv")

