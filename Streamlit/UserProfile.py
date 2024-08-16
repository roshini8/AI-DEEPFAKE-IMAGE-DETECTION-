# Your Streamlit app script
import streamlit as st
import firebase_admin
from firebase_admin import credentials, firestore

def app():
    st.image('Streamlit/NEW.gif', use_column_width=True)
    # Check if the app is not already initialized
    if not firebase_admin._apps:
        cred = credentials.Certificate("Streamlit/key.json")
        firebase_admin.initialize_app(cred)

    db = firestore.client()

    # Main Section
    if st.session_state.get("my_input") == "true":
        st.title("User Profile:")

        email_filter = st.session_state.email

        # Fetch user profile data from Firestore
        user_profiles = db.collection("userprofile").get()

        # Filter user data based on email if provided
        user_data = next((doc.to_dict() for doc in user_profiles if
                          "email" in doc.to_dict() and doc.to_dict()["email"] == email_filter), None)

        if user_data:
            # Display user data
            st.write(f"Gender: {user_data.get('Gender')}")
            st.write(f"Name: {user_data.get('Q1')} {user_data.get('Q2')}")
            st.write(f"Age: {user_data.get('Q3')}")
            st.write(f"Address: {user_data.get('Q4')}")
            st.write(f"Phone Number: {user_data.get('Q5')}")
            st.write(f"Email: {user_data.get('email')}")




        else:
            st.warning("No user data available for the provided email.")

    else:
        st.write("Please log in to access the main page.")







