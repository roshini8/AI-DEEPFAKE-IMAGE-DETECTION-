import streamlit as st
from firebase_admin import auth

def app():
    st.title("Sign Out Page")

    if not st.session_state.get("email"):
        st.write("Please log in to access the sign-out page.")
        return

    if st.session_state.get("signout", False):
        st.text('Name: ' + st.session_state.username)
        st.text('Email id: ' + st.session_state.useremail)
        st.button('Sign out', on_click=sign_out)


def sign_out():
    st.session_state.signout = False
    st.session_state.signedout = False
    st.session_state.username = ''
    st.session_state.email = ''
    st.session_state.my_input = "false"

if __name__ == "__main__":
    app()
