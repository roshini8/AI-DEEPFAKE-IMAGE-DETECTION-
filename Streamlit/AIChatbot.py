import streamlit as st
import json
import nltk
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from streamlit.errors import StreamlitAPIException

# Download NLTK data if not already downloaded
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

# Load chat data from JSON file
def load_json(file):
    with open(file) as bot_responses:
        print(f"Loaded '{file}' successfully!")
        return json.load(bot_responses)

# Load response data
response_data = load_json("Streamlit/chat_data.json")

page_bg_img = f"""
<style>
[data-testid="stAppViewContainer"] > .main {{
background-image: url("https://img.freepik.com/premium-photo/abstract-futuristic-web-with-glowing-neon-light-grid-line-pattern-dark-background_79161-852.jpg?w=1060");
background-size: 100%;
background-position: top left;
background-repeat: no-repeat;
background-attachment: local;
</style>
"""
st.markdown("<h1 style='text-align: center; color: white; font-size: 55px;'>Welcome To Chatbot</h1>",
            unsafe_allow_html=True)
st.markdown(page_bg_img, unsafe_allow_html=True)

# Flag to track if the greeting has been displayed
greeting_displayed = False
# Initialize flags outside of conditional blocks
help_image_section_displayed = False

def display_chat_history(messages):
    # Reverse the messages to display them from bottom to top
    messages.reverse()

    # Display only the last 6 messages
    for i, chat in enumerate(messages[:4]):
        with st.chat_message(chat["role"]):
            st.markdown(chat["content"])

def get_bot_response(user_input):
    def get_response(input_string):
        split_message = word_tokenize(input_string.lower())
        pos_tags = pos_tag(split_message)

        score_list = []
        for response in response_data:
            response_score = 0
            required_score = 0
            required_words = response["required_words"]
            user_input_list = response["user_input"]

            # Check if there are any required words
            if required_words:
                for word, pos_tag_word in pos_tags:
                    if word in required_words:
                        required_score += 1

            # Check if the user input matches the predefined patterns
            if all(word in split_message for word in user_input_list):
                response_score += 1

            # Amount of required words should match the required score
            if required_score == len(required_words):
                score_list.append(response_score)

        best_response = max(score_list, default=0)
        response_index = score_list.index(best_response) if best_response > 0 else -1

        if input_string == "":
            return "Please type something so we can chat :("

        if response_index != -1:
            return response_data[response_index]["bot_response"]
        return "Please try again."

    bot_response = get_response(user_input)
    if bot_response == "Please try again.":
        st.warning(bot_response)  # Display a warning message
    return bot_response

def display_suggestion_buttons():
    col1, col2, col3 = st.columns(3)

    if col1.button("What Is DeepTruth"):
        selected_option = "What Is DeepTruth"
        st.session_state.messages.append({"role": "user", "content": selected_option})
        st.session_state.messages.append(
            {"role": "bot", "content": "DeepTruth is a web application dedicated to swiftly detecting fake images. Utilizing advanced algorithms and deep learning, it empowers users to navigate the digital landscape with confidence by ensuring the authenticity of visual content. Unveil the truth behind the pixels effortlessly with DeepTruth – your reliable ally in the fight against misinformation."})

    if col2.button("How Does DeepTruth Work"):
        selected_option = "How Does DeepTruth Work"
        st.session_state.messages.append({"role": "user", "content": selected_option})
        st.session_state.messages.append(
            {"role": "bot", "content": "DeepTruth operates through a sophisticated combination of machine learning and neural networks. It employs advanced algorithms to meticulously analyze images, scrutinizing pixel patterns and characteristics. Through the power of deep learning, the system continually refines its ability to distinguish between authentic and manipulated images, providing users with a reliable tool to unveil the truth behind visual content in the digital realm."})

    if col3.button("Why Choose DeepTruth"):
        selected_option = "Why Choose DeepTruth"
        st.session_state.messages.append({"role": "user", "content": selected_option})
        st.session_state.messages.append(
            {"role": "bot", "content": "Choose DeepTruth for unparalleled image authenticity verification. Harnessing the prowess of advanced machine learning and neural networks, DeepTruth ensures a cutting-edge approach to detecting fake images. With its commitment to continuous improvement and accuracy, DeepTruth stands as your go-to solution, offering a trustworthy ally in navigating the complex landscape of digital visual content."})

    st.write("")  # Create a new row for the new suggestion buttons

    col4, col5, col6 = st.columns(3)

    if col4.button("Does DeepTruth can detect fake video"):
        selected_option = "Does DeepTruth can detect fake video"
        st.session_state.messages.append({"role": "user", "content": selected_option})
        st.session_state.messages.append(
            {"role": "bot", "content": "Unfortunately, for now DeepTruth couldn't detect any fake video."})

    if col5.button("What type of picture that DeepTruth accept"):
        selected_option = "What type of picture that DeepTruth accept"
        st.session_state.messages.append({"role": "user", "content": selected_option})
        st.session_state.messages.append(
            {"role": "bot", "content": "DeepTruth accept in type of Png, Jpeg and Jpg."})

    # Add a button for the "I need help with my fake images in online!" option
    if col6.button(" I need help with my fake images in online!"):
        selected_option = " I need help with my fake images in online!"
        st.session_state.messages.append({"role": "user", "content": selected_option})
        st.session_state.messages.append(
            {"role": "bot",
             "content": "Please visit this website: https://internetremovals.com/ or stopncii.org. They will help you remove the edited images from all the places on internet. You can also contact MCMC Consumer Care Centre Hotline: 1-800-188-030  "})

    # Add a button for the expert system form
    # Removed the expert system form button

def clear_greeting():
    """Clears the greeting messages from the chat history."""
    global greeting_displayed
    for i, chat in enumerate(st.session_state.messages):
        if "Hello! How can I help you today?" in chat["content"]:
            st.session_state.messages.pop(i)
            greeting_displayed = False
            break

def app():
    # Check for session state
    if st.session_state.get("my_input") == "true":
        if "messages" not in st.session_state:
            st.session_state.messages = []

        global greeting_displayed
        global help_image_section_displayed

        # User input text box
        user_input = st.text_input("Enter your message:")

        try:
            # Greet the chatbot and display buttons only once
            if not greeting_displayed and any(greeting in user_input.lower() for greeting in ["hello", "hi", "good morning"]):
                st.session_state.messages.append({"role": "user", "content": user_input})
                st.session_state.messages.append({"role": "bot", "content": "Hello! How can I help you today?"})

                # Set the flag to True after displaying the greeting
                greeting_displayed = True

            # Display suggestions buttons
            if greeting_displayed:
                display_suggestion_buttons()

            # Check if the "Help me with my image" section should be displayed
            if greeting_displayed and not help_image_section_displayed:
                # New button "Help me with my image"
                if st.button("Help me with my image"):
                    # Set the flag to True to display the section
                    help_image_section_displayed = True

                    # Clear chat history after user clicks "Help me with my image"
                    st.session_state.messages = []
                    clear_greeting()  # Call clear_greeting here



            # Check for form submission and perform actions
            if "form_submitted" not in st.session_state:
                st.session_state.form_submitted = False

            if st.button("Submit Answers") and not st.session_state.form_submitted:
                st.session_state.user_input = user_input  # Store user input in session state
                # For simplicity, let's skip the image classification logic in this example
                st.session_state.messages.append({"role": "user", "content": "Submitted image classification form."})
                st.session_state.messages.append({"role": "bot", "content": "The image is classified. (Result placeholder)"})
                st.session_state.form_submitted = True  # Set the flag to True to indicate form submission

            # Store user input in history on button click
            if st.button("Send"):
                st.session_state.user_input = user_input  # Store user input in session state
                bot_response = get_bot_response(st.session_state.user_input)
                st.session_state.messages.append({"role": "user", "content": st.session_state.user_input})

                # Display bot response only if not a repeated greeting
                if not greeting_displayed:
                    st.session_state.messages.append({"role": "bot", "content": bot_response})

                # Clear the input text box
                st.session_state.user_input = ""

            # Display chat history
            display_chat_history(st.session_state.messages)

        except StreamlitAPIException as e:
            # Catch and ignore the StreamlitAPIException
            pass
    else:
        st.write("Please log in to access the main page.")

if __name__ == "__main__":
    app()
