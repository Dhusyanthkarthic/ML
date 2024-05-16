import streamlit as st
import json
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.tree import DecisionTreeClassifier
from PIL import Image

# Download NLTK resources
import nltk
from nltk.tokenize import word_tokenize

nltk.download('punkt')

# Load background image
background_image = Image.open("stress.jpg")

# Set page configuration
st.set_page_config(
    page_title="Sentiment Analysis",
    page_icon=":bar_chart:",
    layout="wide"
)

# Function to retrieve tweets from file
def retrieve_tweet(data_url):
    tweets_data = []
    tweets_data_path = data_url
    with open(tweets_data_path, "r") as tweets_file:
        for line in tweets_file:
            try:
                tweet = json.loads(line)
                tweets_data.append(tweet)
            except:
                continue
    return tweets_data

# Function to retrieve processed data
def retrieve_processed_data(Pdata_url):
    sent = pd.read_excel(Pdata_url)
    x = []
    y = []
    for i in range(len(tweets_data)):
        if tweets_data[i]['id'] == sent['id'][i]:
            x.append(tweets_data[i]['text'])
            y.append(sent['sentiment'][i])
    return x, y

# Function to train Decision Tree classifier
def dtree_train(x, y):
    vectorizer = CountVectorizer(stop_words='english')
    train_features = vectorizer.fit_transform(x)
    dtree = DecisionTreeClassifier()
    dtree.fit(train_features, [int(r) for r in y])
    return vectorizer, dtree

# Function to classify input tweet using Decision Tree
def classify_tweet(input_tweet, vectorizer, classifier):
    input_features = vectorizer.transform([input_tweet])
    prediction = classifier.predict(input_features)[0]
    return prediction

# Function to find reasons for negative sentiment
def find_reasons(tweet):
    words = word_tokenize(tweet.lower())
    reasons = []
    for word in words:
        if word in ['anxiety', 'anxious', 'anxiously']:
            reasons.append('anxiety')
        elif word in ['sleepless', 'not slept', 'havent slept']:
            reasons.append('sleepless')
        elif word in ['worthless','not good']:
            reasons.append('worthless')
        elif word in ['hopeless','feeling alone','alone']:
            reasons.append('hopeless')
        elif word in ['stressful', 'stressed']:
            reasons.append('stressful')
        elif word in ['worried', 'worries', 'worrying']:
            reasons.append('worry')
        elif word in ['depressed', 'depressing', 'depression']:
            reasons.append('depression')
    return reasons

# Load data and train classifiers
tweets_data = retrieve_tweet('data/tweetdata.txt')
x, y = retrieve_processed_data('processed_data/output.xlsx')
vectorizer_dtree, dtree = dtree_train(x, y)

# Streamlit app
st.title("Sentiment Analysis")
st.markdown(
    """
    <style>
        .css-1vz7ld8 {
            text-align: center;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Set background image
st.image(background_image, use_column_width=True)

# Input text box
input_tweet = st.text_input("Input your tweet:")

# Predict button
if st.button("Predict"):
    if input_tweet:
        prediction = classify_tweet(input_tweet, vectorizer_dtree, dtree)
        if prediction == 1:
            st.write("POSITIVE SENTIMENT!!You are not depressed! It seems like you're feeling positive and balanced. Keep up the good vibes!")
        elif prediction == 0:
            st.write("Neutral sentiment")
        elif prediction == -1:
            st.write("Negative sentiment")

            reasons = find_reasons(input_tweet)
            if reasons:
                st.write("Reasons:")
                for reason in reasons:
                    st.write(f"- {reason}")
            else:
                st.write("No specific reasons found for negative sentiment.")
