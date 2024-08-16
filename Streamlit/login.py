# Your Streamlit app script
import streamlit as st
import firebase_admin
from firebase_admin import credentials, firestore, auth

def app():
    # Check if the app is not already initialized
    if not firebase_admin._apps:
        cred = credentials.Certificate("Streamlit/key.json")
        firebase_admin.initialize_app(cred)

    db = firestore.client()

    # Sign In Section
    st.title("Login")
    email = st.text_input("Email")
    password = st.text_input("Password", type="password")
    submit_button = st.button("Submit")

    # Check login credentials using Firestore
    if submit_button:
        user = None
        try:
            user = auth.get_user_by_email(email)
            # Validate password here if needed
        except auth.AuthError as e:
            st.error(f"Authentication failed: {['email or password may be wrong!']}")
            st.session_state.my_input = "wrong"
        except Exception as e:
            st.error(f"An error occurred: {e}")
            st.session_state.my_input = "wrong"

        if user:
            st.success(f"Welcome, {user.email}!")
            st.session_state.my_input = "true"
            st.session_state.email = email

