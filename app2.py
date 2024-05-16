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
    page_title="Depression Detection",
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
            reasons.append('ANXIETY')
        elif word in ['sleepless', 'not slept', 'havent slept', 'insomnia']:
            reasons.append('SLEEPLESS')
        elif word in ['worthless', 'not good', 'low self-esteem', 'self-loathing']:
            reasons.append('WORTHLESS')
        elif word in ['hopeless', 'feeling alone', 'alone', 'isolated']:
            reasons.append('HOPELESS')
        elif word in ['stressful', 'stressed', 'overwhelmed', 'pressure']:
            reasons.append('STRESSFUL')
        elif word in ['worried', 'worries', 'worrying', 'anxious']:
            reasons.append('WORRY')
        elif word in ['depressed', 'depressing', 'depression', 'sad', 'despair']:
            reasons.append('DEPRESSION')
    return reasons

# Function to suggest treatments based on type of depression
def suggest_treatments(depression_type):
    treatments = {
        'ANXIETY': ["Cognitive Behavioral Therapy (CBT)", "Mindfulness-Based Stress Reduction (MBSR)", "Exposure Therapy"],
        'SLEEPLESS': ["Sleep Hygiene Practices", "Cognitive Behavioral Therapy for Insomnia (CBT-I)", "Relaxation Techniques"],
        'WORTHLESS': ["Cognitive Behavioral Therapy (CBT)", "Interpersonal Therapy (IPT)", "Self-compassion Exercises"],
        'HOPELESS': ["Interpersonal Therapy (IPT)", "Cognitive Behavioral Therapy (CBT)", "Supportive Counseling"],
        'STRESSFUL': ["Stress Management Techniques", "Mindfulness Meditation", "Physical Exercise"],
        'WORRY': ["Cognitive Behavioral Therapy (CBT)", "Mindfulness Meditation", "Worry Time Technique"],
        'DEPRESSION': ["Cognitive Behavioral Therapy (CBT)", "Medication (Antidepressants)", "Exercise Therapy"]
    }
    if depression_type in treatments:
        return treatments[depression_type]
    else:
        return ["Treatment recommendations not available."]

# Streamlit app
st.title("Depression Detection")
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

# Load data and train classifiers
tweets_data = retrieve_tweet('data/tweetdata.txt')
x, y = retrieve_processed_data('processed_data/output.xlsx')
vectorizer_dtree, dtree = dtree_train(x, y)

# Predict button
if st.button("Predict"):
    if input_tweet:
        prediction = classify_tweet(input_tweet, vectorizer_dtree, dtree)
        if prediction == 1:
            st.write("POSITIVE SENTIMENT!!You are not depressed! It seems like you're feeling positive and balanced. Keep up the good vibes!")
        elif prediction == 0:
            st.write("NEUTRAL SENTIMENT!!You're not expressing signs of depression.")
        elif prediction == -1:
            st.write("NEGATIVE SENTIMENT!!!")
            reasons = find_reasons(input_tweet)
            if reasons:
                st.subheader("Possible Reasons for Negative Sentiment:")
                for reason in reasons:
                    st.write(f"- {reason}")
                st.subheader("Recommended Treatments:")
                for reason in reasons:
                    treatments = suggest_treatments(reason)
                    st.write(f"Treatments for {reason}:")
                    for treatment in treatments:
                        st.write(f"- {treatment}")
            else:
                st.write("No specific reasons found for negative sentiment.")
            st.subheader("Mental Health Tips")
            st.write("""
            - Practice mindfulness and deep breathing exercises daily.
            - Stay connected with friends and family members for emotional support.
            - Engage in regular physical activity to boost your mood.
            - Limit your exposure to negative news and social media content.
            - Seek professional help if you're feeling overwhelmed or stressed.
            """)
