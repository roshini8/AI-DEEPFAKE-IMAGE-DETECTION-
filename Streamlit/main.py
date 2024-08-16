import json
import streamlit as st
from streamlit_lottie import st_lottie

if "my_input" not in st.session_state:
    st.session_state.my_input = ""

my_input = st.session_state.my_input
if my_input == "true":
    # Main page content goes here
    st.title("Main Page")
    st.write("Welcome to the main page!")
else:
    # Display login form if the user is not logged in
    st.write("Please log in to access the main page.")