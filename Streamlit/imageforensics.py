import streamlit as st
from face_auth import app as face_auth_app
from detector import app as detector_app

def app():


        # Set up the app title
        st.title("Forgery Detection App")

        # Create a dropdown for selecting the page
        selected_page = st.selectbox("Select Page", ["Face Forgery Detection", "Image Forgery Detection"])

        # Display the selected page
        if selected_page == "Face Forgery Detection":
            face_auth_app()
        elif selected_page == "Image Forgery Detection":
            detector_app()


