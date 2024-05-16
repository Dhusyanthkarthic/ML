import streamlit as st
import json
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.tree import DecisionTreeClassifier
from PIL import Image

# Load background image
background_image = Image.open("stress.jpg")

# Set page configuration
st.set_page_config(
    page_title="DEPRESSION DETECTION",
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

# Load data and train classifiers
tweets_data = retrieve_tweet('data/tweetdata.txt')
x, y = retrieve_processed_data('processed_data/output.xlsx')
vectorizer_dtree, dtree = dtree_train(x, y)

# Streamlit app
st.title("DEPRESSION DETECTION")
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

if st.button("Predict"):
    print("Predict button clicked")  # Debugging statement
    input_tweet = st.text_input("Input your tweet:")
    if input_tweet:
        print("Input tweet received:", input_tweet)  # Debugging statement
        prediction = classify_tweet(input_tweet, vectorizer_dtree, dtree)
        print("Prediction:", prediction)  # Debugging statement
        if prediction == 1:
            sentiment = "Positive"
            st.write("You are not depressed! It seems like you're feeling positive and balanced. Keep up the good vibes!")

        elif prediction == 0:
            sentiment = "Neutral"
        elif prediction == -1:
            sentiment = "Negative"
        else:
            sentiment = "Unknown"
        print("Predicted sentiment:", sentiment)  # Debugging statement
        st.write(f"The sentiment of the input tweet is: {sentiment}")
        
        # Display tips and available treatments only when prediction is -1
        if prediction == -1:
            st.subheader("Mental Health Tips")
            st.write("""
            - Practice mindfulness and deep breathing exercises daily.
            - Stay connected with friends and family members for emotional support.
            - Engage in regular physical activity to boost your mood.
            - Limit your exposure to negative news and social media content.
            - Seek professional help if you're feeling overwhelmed or stressed.
            """)

            st.subheader("Available Treatments")
            st.write("""
            - Cognitive Behavioral Therapy (CBT)
            - Medication (Antidepressants, Anxiolytics)
            - Exercise Therapy
            - Mindfulness-Based Stress Reduction (MBSR)
            - Support Groups
            """)
