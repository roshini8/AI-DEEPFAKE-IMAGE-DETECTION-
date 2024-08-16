import streamlit as st
from AIChatbot import app as chatbot_app
from chat import app as chat_app

def app():
    # Set up the app title
    st.title("AI Assistant")

    # Create a dropdown for selecting the page
    selected_page = st.selectbox("Select Page", ["Text Based Assistant", "Voice Based Assistant"])

    # Display the selected page
    if selected_page == "Text Based Assistant":
        chatbot_app()
    elif selected_page == "Voice Based Assistant":
        chat_app()
