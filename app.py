import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt

st.markdown("## Hey!, this is place where you can Analysis top 10 twitter posts")
st.sidebar.markdown("Training Data Visulization")

data_path = ("/Users/anjalisingh/Downloads/Streamlit/tweets.csv")

@st.cache(persist=True)
def load_data():
    data = pd.read_csv(data_path)
    return data


data = load_data()

st.sidebar.markdown("### Number of tweets")
select = st.sidebar.selectbox('Visualization Type',['Histogram','PieChart'])

sentiment_count = data['label'].value_counts()
sentiment_count = pd.DataFrame({'Sentiments':sentiment_count.index,'Tweets':sentiment_count.values})

if st.sidebar.checkbox('Show',False,key='0'):
    st.markdown("### No. of tweets by sentiments ")
    if select=='Histogram':
        fig = px.bar(sentiment_count,x='Sentiments',y='Tweets',color='Tweets',height=500)
        st.plotly_chart(fig)
    else:
        fig = px.pie(sentiment_count,values='Tweets',names='Sentiments')
        st.plotly_chart(fig)

    # Word cloud
st.sidebar.subheader("Word Cloud")
label = [l for l in pd.unique(data["label"]) if pd.isnull(l)==False]
print(label)
word_sentiment = st.sidebar.radio("Which Sentiment to Display?", label)
if st.sidebar.checkbox("Show", False, key="6"):
    # to remove a pyplot warning
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.subheader(f"Word Cloud for {word_sentiment.capitalize()} Sentiment")
    df = data[data["label"]==word_sentiment]
    words = " ".join(df["full_text"])
    processed_words = " ".join([word for word in words.split() if "http" not in word and not word.startswith("@") and word != "RT"])
    wordcloud = WordCloud(stopwords=STOPWORDS, background_color="white", width=800, height=640).generate(processed_words)
    plt.imshow(wordcloud)
    plt.xticks([])
    plt.yticks([])
    st.pyplot()

st.write("# Twitter Post Sentiments Analysis")
message_text = st.text_input("Search from twitter")

import joblib
import pickle
from get_tweets import get_related_tweets
from tensorflow.keras.preprocessing.sequence import pad_sequences

pipeline = joblib.load('cb_sgd_final.sav')

def requestResults(name):
    tweets = get_related_tweets(name)
    tokenizer = pickle.load(open('tokenizer1.pkl', 'rb'))
    tokenizer.fit_on_texts(tweets["tweet_text"])
    word_index = tokenizer.word_index
    sequence = tokenizer.texts_to_sequences(tweets["tweet_text"])
    padded=pad_sequences(sequence, maxlen=38790)
    tweets['prediction'] = pipeline.predict(padded)
    data = str(tweets.prediction.value_counts()) + '\n\n'
    print(tweets['tweet_text'])
    return tweets

if message_text != '':
  result =  requestResults(message_text)
  for i in range(0, len(result['prediction'])):
    if result['prediction'][i]==0:
        result['prediction'][i]="Non-offensive"
    else:
        result['prediction'][i]="Offensive"
  st.write(result)
  st.set_option('deprecation.showPyplotGlobalUse', False)
  st.sidebar.markdown("Twitter Data Visulization")
  st.sidebar.subheader("Word Cloud")
  word_sentiment = st.sidebar.radio("Which Sentiment to Display?", tuple(pd.unique(result['prediction'])))
  if st.sidebar.checkbox("Show", False, key="10"):
    st.subheader(f"Word Cloud for {word_sentiment} Sentiment")
    df = result[result["prediction"]==word_sentiment]
    words = " ".join(df['tweet_text'])
    processed_words = " ".join([word for word in words.split() if "http" not in word and not word.startswith("@") and word != "RT"])
    wordcloud = WordCloud(stopwords=STOPWORDS, background_color="white", width=800, height=640).generate(processed_words)
    plt.imshow(wordcloud)
    plt.xticks([])
    plt.yticks([])
    st.pyplot()



