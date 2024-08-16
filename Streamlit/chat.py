# import libraries
import speech_recognition as sr  # recognise speech
import random
from time import ctime  # get time details
import webbrowser  # open browser
import pyttsx3
import streamlit as st

# Initialization functions
user = 'Siri'
asis = 'Friend'

def setUserName(name):
    global user
    user = name

def setAsistantName(name):
    global asis
    asis = name

def there_exists(terms, voice_data):
    for term in terms:
        if term in voice_data:
            return True
    return False

# listen for audio and convert it to text:
def record_audio(ask=""):
    # Listening functions
    r = sr.Recognizer()  # initialize a recognizer
    with sr.Microphone() as source:  # microphone as source
        if ask:
            print("engine is", ask)
            engine_speak(ask)
            print("start listening")
        try:
            audio = r.listen(source, timeout=10, phrase_time_limit=10)  # listen for the audio via source
            print("Done Listening")
            voice_data = r.recognize_google(audio)  # convert audio to text
            print(">>", voice_data.lower())  # print what the user said
            return voice_data.lower()
        except sr.UnknownValueError:  # error: recognizer does not understand
            engine_speak('I did not get that')
        except sr.RequestError:
            engine_speak('Sorry, the service is down')  # error: recognizer is not connected
        except sr.WaitTimeoutError:
            engine_speak("Listening timed out. Please try again.")
            return "Listening timed out. Please try again."


# Speaking Functions

# Step1: convert speech to text
def engine_speak(text):
    engine = pyttsx3.init()
    print(text + " .....")
    text = str(text)
    print("must speak now!")
    engine.say(text)  # convert text to audio
    engine.runAndWait()  # speak the audio through speakers

# Step2: Response functionality
def respond(voice_data):
    # 1: greeting
    if there_exists(['hey', 'hi', 'hello'], voice_data):
        greetings = ["hey, how can I help you" + user, "hey, what's up?" + user, "I'm listening" + user,
                     "how can I help you?" + user, "hello" + user]
        greet = greetings[random.randint(0, len(greetings) - 1)]
        engine_speak(greet)

    # 2: assistant name
    if there_exists(["what is your name", "what's your name", "tell me your name"], voice_data):

        if user:
            engine_speak(f"My name is {asis}, {user}")  # gets users name from voice input
        else:
            engine_speak(f"My name is {asis}. what's your name?")  # in case you haven't provided your name.

    if there_exists(["your name should be"], voice_data):
        asis_name = voice_data.split("be")[-1].strip()
        engine_speak("okay, I will remember that my name is " + asis_name)
        setAsistantName(asis_name)  # remember name in asis variable

    if there_exists(["my name is"], voice_data):
        person_name = voice_data.split("is")[-1].strip()
        engine_speak("okay, I will remember that " + person_name)
        setUserName(person_name)  # remember name in person variable

    if there_exists(["what is my name"], voice_data):
        engine_speak("Your name must be " + user)

    # 3: greeting
    if there_exists(["how are you", "how are you doing"], voice_data):
        engine_speak("I'm very well, thanks for asking " + user)

    if there_exists(["How"], voice_data):
        engine_speak(
            "DeepTruth uses thousands of images to create a model using Convolutional Neural Network and Analyses the images using Heatmap and Error Level Analysis.")


    # 4: time
    if there_exists(["what's the time", "tell me the time", "what time is it", "what is the time"], voice_data):
        time = ctime().split(" ")[4].split(":")[0:2]
        if time[0] == "00":
            hours = '12'
        else:
            hours = time[0]
        minutes = time[1]
        time = hours + " and " + minutes + "minutes"
        engine_speak("The time is " + time)

    # 5: search google
    if there_exists(["search for"], voice_data) and 'on youtube' not in voice_data:
        search_term = voice_data.split("for")[-1]
        url = "https://google.com/search?q=" + search_term
        webbrowser.get().open(url)
        engine_speak("Here is what I found for" + search_term + "on google")

    # 6: search youtube
    if there_exists(["on youtube"], voice_data):
        search_term = voice_data.split("for")[-1]
        search_term = search_term.replace("on youtube", "").replace("search", "")
        url = "https://www.youtube.com/results?search_query=" + search_term
        webbrowser.get().open(url)
        engine_speak("Here is what I found for " + search_term + "on youtube")

    # 7: get stock price
    if there_exists(["Deepfake"], voice_data):
        url = "https://en.wikipedia.org/wiki/Deepfake"
        webbrowser.get().open(url)
        engine_speak("Here is what I found for you on google")

    # 8 weather
    if there_exists(["weather"], voice_data):
        url = "https://www.google.com/search?sxsrf=ACYBGNSQwMLDByBwdVFIUCbQqya-ET7AAA%3A1578847393212&ei=oUwbXtbXDN-C4-EP-5u82AE&q=weather&oq=weather&gs_l=psy-ab.3..35i39i285i70i256j0i67l4j0i131i67j0i131j0i67l2j0.1630.4591..5475...1.2..2.322.1659.9j5j0j1......0....1..gws-wiz.....10..0i71j35i39j35i362i39._5eSPD47bv8&ved=0ahUKEwiWrJvwwP7mAhVfwTgGHfsNDxsQ4dUDCAs&uact=5"
        webbrowser.get().open(url)
        engine_speak("Here is what I found for on google")

    # 9: Information about DeepTruth
    if there_exists(["tell me about DeepTruth", "what is DeepTruth", "explain DeepTruth"], voice_data):
        deeptruth_info = "DeepTruth is an innovative platform that leverages advanced technologies like natural language processing and machine learning to provide intelligent responses and perform various tasks. It is designed to assist users in obtaining information, performing searches, and interacting using voice commands."
        engine_speak(deeptruth_info)

    if there_exists(["DeepTruth,choose,Why"], voice_data):
        engine_speak(
            "Choose DeepTruth for unparalleled image authenticity verification. Harnessing the prowess of advanced machine learning and neural networks, DeepTruth ensures a cutting-edge approach to detecting fake images. With its commitment to continuous improvement and accuracy, DeepTruth stands as your go-to solution, offering a trustworthy ally in navigating the complex landscape of digital visual content.")

    if there_exists(["What Is"], voice_data):
        engine_speak(
            "DeepTruth is a web application dedicated to swiftly detecting fake images. Utilizing advanced algorithms and deep learning, it empowers users to navigate the digital landscape with confidence by ensuring the authenticity of visual content. Unveil the truth behind the pixels effortlessly with DeepTruth – your reliable ally in the fight against misinformation.")


    if there_exists(["exit", "quit", "bye", "done"], voice_data):
        engine_speak("bye")
        return 0

    if there_exists(["Help"], voice_data):
        url = "https://report.stopncii.org/case/create?lang=en-gb"
        webbrowser.get().open(url)
        engine_speak(
            "Here is what deeptruth finds a solution for you")
    return 1

def app():
    # Check for session state
    if st.session_state.get("my_input") == "true":

        setAsistantName("Assistant")
        setUserName("User")

        # Add a welcome message or title
        engine_speak("Welcome to Voice Based Assistant!")

        while True:
            print("Listening...")
            voice_data = record_audio("How can I assist you?")

            if there_exists(["exit", "quit", "bye", "done"], voice_data):
                break

            response_flag = respond(voice_data)

            if response_flag:
                print("Listening for the next command...")
    else:
        st.write("Please log in to access the main page.")

if __name__ == "__main__":
    app()
