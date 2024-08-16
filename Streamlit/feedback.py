import streamlit as st
import firebase_admin
from firebase_admin import credentials, firestore, initialize_app, auth
from transformers import pipeline
import pandas as pd
from streamlit_star_rating import st_star_rating
import matplotlib.pyplot as plt
from io import BytesIO

# Check if the app is not already initialized
if not firebase_admin._apps:
    cred = credentials.Certificate("Streamlit/key.json")
    initialize_app(cred)

db = firestore.client()
nlp = pipeline("sentiment-analysis")


def analyze_sentiment(text):
    result = nlp(text)
    return result[0]['label'], result[0]['score']


def get_color(sentiment_label):
    if sentiment_label == 'POSITIVE':
        return 'green'  # Positive sentiment is green
    else:
        return 'red'  # Negative sentiment is red


def app():
    # Check if the app is not already initialized
    if st.session_state.get("my_input") == "true":
        email = st.session_state.email
        st.title("Feedback Collection")

        st.subheader("Rate Your Experience")
        stars = st_star_rating("Please rate your experience", maxValue=5, defaultValue=3, key="rating")

        st.subheader("Feedback Questions")
        q1 = st.text_input("Q1: What did you like the most?")
        q2 = st.text_input("Q2: What could be improved?")
        q3 = st.text_input("Q3: Any additional comments or suggestions?")

        submit_button = st.button("Submit Feedback")

        if submit_button:
            feedback_data = {
                "email": email,
                "Rating": stars,
                "Liked_Most": q1,
                "Improvement_Suggestions": q2,
                "Additional_Comments": q3,
            }

            sentiment_label, sentiment_score = analyze_sentiment(q1 + " " + q2 + " " + q3)
            feedback_data["Sentiment_Label"] = sentiment_label
            feedback_data["Sentiment_Score"] = sentiment_score

            db.collection("feedback").add(feedback_data)
            st.success("Thank you for your feedback!")

            # Display sentiment analysis results in a custom bar chart
            st.subheader("Sentiment Analysis Results")

            # Get color based on sentiment score
            sentiment_color = get_color(sentiment_label)

            fig, ax = plt.subplots(figsize=(6, 4))  # Adjust the figsize as needed
            bar_height = [max(0, sentiment_score), 0]  # Set negative sentiment to zero
            bar_labels = [sentiment_label, ""]
            bars = ax.bar(bar_labels, bar_height, color=sentiment_color, label="Sentiment")
            ax.set_ylabel('Sentiment Score')
            ax.set_title('Sentiment Analysis Results')

            # Manually set y-axis limits to start from 0.00
            ax.set_ylim(0, 1)

            # Add legend
            ax.legend()

            # Convert the plot to PNG image
            image_stream = BytesIO()
            plt.savefig(image_stream, format='png')
            plt.close(fig)

            # Display the PNG image in Streamlit
            st.image(image_stream.getvalue(), use_column_width=True)

            # Display sentiment score below the bar chart
            st.write(f"Sentiment Score: {sentiment_score:.4f}")

    else:
        st.write("Please log in to access the main page.")
