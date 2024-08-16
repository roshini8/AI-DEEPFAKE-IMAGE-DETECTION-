import streamlit as st
import tensorflow as tf
from keras.applications import ResNet50
from sklearn.ensemble import IsolationForest
from PIL import Image
import numpy as np
import pandas as pd  # Added import for Pandas
import matplotlib.pyplot as plt
from firebase_admin import firestore, credentials, initialize_app



# Load pre-trained model and classifier
model = ResNet50(weights='imagenet')

# Function to load and preprocess the image
def load_image(image_file):
    img = Image.open(image_file).resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Function to get predictions from the deepfake detection model
def predict_image(img_array):
    global model
    prediction = model.predict(img_array)
    return np.argmax(prediction), np.max(prediction)

# Function to display the image and its corresponding deepfake anomaly score
def display_image_and_anomaly_score(image, anomaly_score):
    plt.subplot(121)
    plt.imshow(image.squeeze())
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(122)
    plt.plot([anomaly_score])
    plt.title("Deepfake Anomaly Score")
    plt.xlabel("Image")
    plt.ylabel("Anomaly Score")
    st.pyplot(plt)

# Function to train Isolation Forest model
def train_isolation_forest(data):
    isolation_model = IsolationForest(contamination=0.1, random_state=42)
    data['AnomalyScore'] = isolation_model.fit_predict(data[['Confidences_real', 'Confidences_fake']])
    return data, isolation_model

# Function to fetch data from Firestore
def fetch_data_from_firestore(db):
    # Fetch data from Firestore as needed
    users = db.collection("userprofile").stream()
    data = {'Confidences_real': [], 'Confidences_fake': []}
    for user in users:
        predictions = db.collection("userprofile").document(user.id).collection("image_results").stream()
        for prediction in predictions:
            data['Confidences_real'].append(prediction.to_dict().get('Confidences_real', 0))
            data['Confidences_fake'].append(prediction.to_dict().get('Confidences_fake', 0))
    return pd.DataFrame(data)  # Convert data to a Pandas DataFrame

# Streamlit app
def app():
    st.title('Deepfake Detection and Anomaly Detection System')
    uploaded_file = st.file_uploader('Choose an image...', type=['jpg', 'jpeg', 'png'])

    # Initialize Firestore client
    db = firestore.client()

    # Load data from Firestore for anomaly detection
    data = fetch_data_from_firestore(db)

    # Train Isolation Forest model
    data, isolation_model = train_isolation_forest(data)

    if uploaded_file is not None:
        image = load_image(uploaded_file)
        label, confidence = predict_image(image)
        if label == 0:
            result = 'Real'
        else:
            result = 'Deepfake'

        st.write(f'Prediction: {result}')
        st.write(f'Confidence: {confidence * 100:.2f}%')

        # Calculate anomaly score using Isolation Forest
        anomaly_score = isolation_model.decision_function(np.array([[data['Confidences_real'].iloc[-1], data['Confidences_fake'].iloc[-1]]]))

        # Display the image and its corresponding deepfake anomaly score
        display_image_and_anomaly_score(image, anomaly_score[0])

# Run the Streamlit app
if __name__ == '__main__':
    app()
