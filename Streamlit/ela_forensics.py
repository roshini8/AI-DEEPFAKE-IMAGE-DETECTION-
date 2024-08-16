import streamlit as st
from PIL import Image, ImageChops, ImageEnhance
import numpy as np
from keras.models import load_model
import cv2
# Prediction module
def prepare_image(uploaded_file):
    image_size = (128, 128)
    return (
        np.array(convert_to_ela_image(uploaded_file, 90).resize(image_size)).flatten()
        / 255.0
    )

def predict_result(uploaded_file):
    model = load_model("C:/Users/roshi/PycharmProjects/DEEPTRUTH/Streamlit/finaldfd.h5")
    class_names = ["Forged", "Authentic"]
    test_image = prepare_image(uploaded_file)
    test_image = test_image.reshape(-1, 128, 128, 3)

    y_pred = model.predict(test_image)
    y_pred_class = round(y_pred[0][0])

    prediction = class_names[y_pred_class]
    if y_pred <= 0.5:
        confidence = f"{(1-(y_pred[0][0])) * 100:0.2f}"
    else:
        confidence = f"{(y_pred[0][0]) * 100:0.2f}"
    return prediction, confidence
def detect_faces(image):
    face_cascade = cv2.CascadeClassifier('Streamlit/haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)

    return image, faces

# ELA module
import tempfile
import os

def convert_to_ela_image(uploaded_file, quality):
    original_image = Image.open(uploaded_file).convert("RGB")

    # Create a temporary file to save the image
    temp_file_path = os.path.join(tempfile.gettempdir(), "temp_image.jpg")
    original_image.save(temp_file_path, "JPEG", quality=quality)

    resaved_image = Image.open(temp_file_path)

    ela_image = ImageChops.difference(original_image, resaved_image)

    extrema = ela_image.getextrema()
    max_difference = max([pix[1] for pix in extrema])
    if max_difference == 0:
        max_difference = 1
    scale = 350.0 / max_difference

    ela_image = ImageEnhance.Brightness(ela_image).enhance(scale)

    ela_image.save("ela_image.png")

    # Remove the temporary file
    os.remove(temp_file_path)

    return ela_image

# Streamlit app
def app():
    st.title("Image Analysis App")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        original_image = Image.open(uploaded_file)
        # Convert images to np.uint8
        image_np = np.array(original_image).astype(np.uint8)
        # Detect faces
        result_img, result_faces = detect_faces(image_np)

        st.subheader("Original Image:")
        st.image(original_image, use_column_width=True)

        # Display ELA image
        ela_image = convert_to_ela_image(uploaded_file, 90)
        st.subheader("ELA Image:")
        st.image(ela_image, use_column_width=True)

        if st.button("Analyze Image"):
            prediction, confidence = predict_result(uploaded_file)
            st.subheader("Result:")
            st.write(f"Prediction: {prediction}\nConfidence: {confidence} %")

if __name__ == "__main__":
    app()
