import streamlit as st
import home
import detector
#import login
import signup
import feedback
import UserProfile
import Logout
import AIChatbot
import anomaly_detection
import face_auth
import ela_forensics
import imageforensics
import json
from streamlit_lottie import st_lottie
from firebase_admin import credentials, auth, firestore, initialize_app
import firebase_admin
import chat
import Assistant

# Check if the app is not already initialized
if not firebase_admin._apps:
    cred = credentials.Certificate("Streamlit/key.json")
    initialize_app(cred)

db = firestore.client()

global greeting_displayed
global help_image_section_displayed

st.set_page_config(
    page_title="Deeptruth",
    page_icon="🤖",
    layout="wide")



PAGES = {
    "Dashboard": home,
    "Login/Signup": signup,
    "Feedback": feedback,
    #"ELA": ela_forensics,
    "Image Forensics": imageforensics,
    #"Fake Image Detection": detector,
    #"DeepFake": face_auth,
    "Chatbot": Assistant,
    #"Anomaly Detector": anomaly_detection,
    "Signout": Logout

}

st.sidebar.title("Deeptruth")

st.sidebar.write(
    "Deeptruth is a tool that utilizes the power of Deep Learning to distinguish Real images from the Fake ones.")

st.sidebar.subheader('Navigation:')
selection = st.sidebar.radio("", list(PAGES.keys()))

page = PAGES[selection]

page.app()

