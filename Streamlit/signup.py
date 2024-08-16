import streamlit as st
from firebase_admin import credentials, auth, firestore, initialize_app
import firebase_admin

# Check if the app is not already initialized
if not firebase_admin._apps:
    cred = credentials.Certificate("Streamlit/key.json")
    initialize_app(cred)

db = firestore.client()

def app():
    st.image('Streamlit/NEW.gif', use_column_width=True)

    st.title("User Profile")

    st.markdown('<h1 style="color: skyblue; text-shadow: 2px 2px 5px #0000FF;">[DeepTruth] 😎</h1>', unsafe_allow_html=True)

    if 'username' not in st.session_state:
        st.session_state.username = ''
    if 'useremail' not in st.session_state:
        st.session_state.useremail = ''

    def f():
        try:
            user = auth.get_user_by_email(email)
            st.session_state.username = user.uid
            st.session_state.useremail = user.email

            global Usernm
            Usernm = (user.uid)

            st.session_state.signedout = True
            st.session_state.signout = True

            # Set the 'email' attribute in st.session_state
            st.session_state.email = user.email

            # Redirect to user profile page
            st.session_state.my_input = "true"

        except:
            st.warning('Login Failed')

    def t():
        st.session_state.signout = False
        st.session_state.signedout = False
        st.session_state.username = ''
        st.session_state.email = ''  # Reset the email attribute

    if "signedout" not in st.session_state:
        st.session_state["signedout"] = False
    if 'signout' not in st.session_state:
        st.session_state['signout'] = False

    if not st.session_state["signedout"]:
        # only show if the state is False, hence the button has never been clicked
        choice = st.selectbox('Login/Signup', ['Login', 'Sign up'])
        email = st.text_input('Email Address')
        password = st.text_input('Password', type='password')

        if choice == 'Sign up':
            username = st.text_input("Enter your unique username")
            q1 = st.text_input("Please enter your first name")
            q2 = st.text_input("Please enter your last name")
            q3 = st.text_input("Please enter your age")
            q4 = st.text_input("Please enter your address")
            q5 = st.text_input("Please enter your phone number")
            gender_options = ["Male", "Female"]
            gender = st.selectbox("Please select your gender", options=gender_options)

            if st.button('Create my account'):
                # Set the 'uid' field to the entered username
                user = auth.create_user(email=email, password=password, uid=username)

                st.success('Account created successfully!')
                st.markdown('Please Login using your email and password')
                st.balloons()

                # Store user profile in Firestore using the username as the document name
                user_profile = {
                    "email": email,
                    "Q1": q1,
                    "Q2": q2,
                    "Q3": q3,
                    "Q4": q4,
                    "Q5": q5,
                    "Gender": gender,
                }
                db.collection("userprofile").document(username).set(user_profile)

        else:
            # st.button('Login', on_click=f)
            st.button('Login', on_click=f)

    if st.session_state.get("signout", False):  # Check if the key 'signout' is present
        # Display user details instead of sign-out button
        user_profiles = db.collection("userprofile").document(st.session_state.username).get()
        user_data = user_profiles.to_dict()

        if user_data:
            st.write(f"Gender: {user_data.get('Gender')}")
            st.write(f"Name: {user_data.get('Q1')} {user_data.get('Q2')}")
            st.write(f"Age: {user_data.get('Q3')}")
            st.write(f"Address: {user_data.get('Q4')}")
            st.write(f"Phone Number: {user_data.get('Q5')}")
            st.write(f"Email: {st.session_state.useremail}")
        else:
            st.warning("No user data available.")

    # Add this line to let other modules open
    st.session_state.my_input = "true" if st.session_state.get("email") else "false"



if __name__ == "__main__":
    app()
