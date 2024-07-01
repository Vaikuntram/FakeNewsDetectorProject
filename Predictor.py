import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from scipy.sparse import hstack
import streamlit as st
import re

df = pd.read_csv('FakeNewsNet.csv')

df.dropna(axis=0,inplace=True)

y = df['real']

X_title = df['title']
X_newsurl = df['news_url']
X_source = df['source_domain']

x_train_title, x_test_title, x_train_url, x_test_url, x_train_domain, x_test_domain, y_train, y_test = train_test_split(
    X_title, X_newsurl, X_source, y, test_size=0.25, random_state=42)

vectorizer_title = CountVectorizer()
vectorizer_url = CountVectorizer()
vectorizer_source = CountVectorizer()

Xv_title_train = vectorizer_title.fit_transform(x_train_title)
Xv_newsurl_train = vectorizer_url.fit_transform(x_train_url)
Xv_source_train = vectorizer_source.fit_transform(x_train_domain)

Xv_title_test = vectorizer_title.transform(x_test_title)
Xv_newsurl_test = vectorizer_url.transform(x_test_url)
Xv_source_test = vectorizer_source.transform(x_test_domain)

Xv_train = hstack([Xv_title_train,Xv_newsurl_train,Xv_source_train])
Xv_test = hstack([Xv_title_test,Xv_newsurl_test,Xv_source_test])

model = LogisticRegression()
model.fit(Xv_train,y_train)

train_score = model.score(Xv_train,y_train)
test_score = model.score(Xv_test,y_test)

print(train_score)
print(test_score)

res = model.predict(Xv_test)
print(res)

st.title('Fake News Detector')

st.header('Please enter the following details:')

title = st.text_input('Title of the article')
url = st.text_input('URL of the article')
source = st.text_input('Source Domain of the article')

sample_title = [title]
sample_url = [url]
sample_source = [source]

sample_title_vect = vectorizer_title.transform(sample_title)
sample_newsurl_vect = vectorizer_url.transform(sample_url)
sample_source_vect = vectorizer_source.transform(sample_source)

sample_vect = hstack([sample_title_vect, sample_newsurl_vect, sample_source_vect])

pred = model.predict(sample_vect)
prob = model.predict_proba(sample_vect)

print(pred)
print(prob)

if st.button('Enter'):
    if(pred[0] == 1):
        st.subheader('This article is most likely real')
        st.subheader("There's a "+str(prob[0][1]*100)+"% chance that this article is real")
    else:
        st.subheader("This article is most likely fake")
        st.subheader("There's a "+str(prob[0][0]*100)+"% chance that this article is fake")

