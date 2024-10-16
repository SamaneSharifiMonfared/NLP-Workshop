import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from scipy.special import softmax
plt.style.use('ggplot')
import nltk
import plotly.express as px


df_sorted = pd.read_csv("output/sorted.csv")


fig = px.histogram(df_sorted, x='score', template='plotly_white', title='Sentiment of Answers based on length')
fig.show()

print(df_sorted)

