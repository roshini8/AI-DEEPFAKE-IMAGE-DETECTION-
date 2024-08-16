import streamlit as st
import util
import base64
import firebase_admin
from firebase_admin import credentials, firestore,storage
from PIL import Image
import numpy as np
import cv2
from PIL import Image, ImageChops, ImageEnhance
import pdfkit
import jinja2
import base64
import os
from datetime import datetime

# Check if the app is not already initialized
if not firebase_admin._apps:
    cred = credentials.Certificate("Streamlit/key.json")
    firebase_admin.initialize_app(cred)

db = firestore.client()

def detect_faces(image):
    face_cascade = cv2.CascadeClassifier('Streamlit/haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)

    return image, faces

# Add a new function to calculate ELA and threshold
def calculate_ela_image(file_uploaded):
    # Open the image using PIL
    original_image = Image.open(file_uploaded)

    # Save the image in a temporary file
    temp_filename = "temp.jpg"
    original_image.save(temp_filename, quality=90)

    # Open the saved image
    temp_image = Image.open(temp_filename)

    # Calculate ELA
    ela = ImageChops.difference(original_image, temp_image)
    extrema = ela.getextrema()
    max_diff = max([ex[1] for ex in extrema])
    scale = 255.0 / max_diff

    ela = ImageEnhance.Brightness(ela).enhance(scale)

    # Remove the temporary file
    os.remove(temp_filename)

    return ela, max_diff

def rep_gen(image, prediction, probability, ela_img, max_diff):
    # Encode image, ela_img to Base64
    encoded_image = cv2.imencode('.jpg', image)[1].tobytes()
    original_image_base64 = base64.b64encode(encoded_image).decode('utf-8')

    ela_img = np.array(ela_img)
    ela_img = cv2.imencode('.jpg', ela_img)[1].tobytes()
    ela_img_base64 = base64.b64encode(ela_img).decode('utf-8')

    context = {
        'prediction': prediction,
        'confidences': probability,
        'ela_img': ela_img_base64,
        'ela_threshold': max_diff,
        'original_image': original_image_base64,
    }

    template_loader = jinja2.FileSystemLoader('./')
    template_env = jinja2.Environment(loader=template_loader)
    template = template_env.get_template("Streamlit/rep_gen.html")
    output = template.render(context)
    config = pdfkit.configuration(wkhtmltopdf='C:\\Program Files\\wkhtmltopdf\\bin\\wkhtmltopdf.exe')

    pdf_filename = f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    pdf_path = os.path.join('./', pdf_filename)
    pdfkit.from_string(output, pdf_path, configuration=config)

    # Upload PDF to Firebase Storage
    upload_pdf_to_storage(pdf_path, pdf_filename)

def upload_pdf_to_storage(pdf_path, pdf_filename):
    """Upload the PDF to Firebase Storage"""
    bucket = storage.bucket('myapplication-71465.appspot.com')  # Specify your storage bucket name here
    blob = bucket.blob(f"pdfs/{pdf_filename}")
    blob.upload_from_filename(pdf_path)

def app():
    st.title("Fake Image Forgery Detection")
    st.write("Upload a Picture to see if it is a fake or real image.")
    st.markdown('*Need a face to test? Visit this [link]("https://github.com/kanakmi/Deforgify/tree/main/Model%20Training/dataset")*')
    file_uploaded = st.file_uploader("Choose the Image File", type=["jpg", "png", "jpeg"])

    if file_uploaded is not None:
        # Open the image using PIL
        image = Image.open(file_uploaded)

        # Convert images to np.uint8
        image_np = np.array(image).astype(np.uint8)

        # Detect faces
        result_img, result_faces = detect_faces(image_np)

        # Resize the image to 128x128
        result_img_resized = cv2.resize(result_img, (128, 128))

        # Display resized original image with face detection in column c1
        c1, buff, c2 = st.columns([2, 0.5, 2])

        st.success("Found {} faces\n".format(len(result_faces)))

        # Classify the image
        res = util.classify_image(file_uploaded)

        # Display the classification result along with metadata analysis in column c2

        c2.subheader("Classification Result")
        c1.image(result_img_resized, use_column_width=True)
        c2.write("The image is classified as **{}**.".format(res['label'].title()))
        c2.write("Fake Confidence level: **{}%**.".format(res['probability']))

        # Generate and display metadata analysis
        image_format = image.format
        image_mode = image.mode
        image_size = image.size

        c2.subheader("Metadata Analysis")
        c2.write(f"Image Format: {image_format}")
        c2.write(f"Image Mode: {image_mode}")
        c2.write(f"Image Size: {image_size}")

        # Error Level Analysis (ELA)
        c1.subheader("Error Level Analysis (ELA)")
        ela_img, ela_threshold = calculate_ela_image(file_uploaded)
        c1.image(ela_img, caption="ELA", use_column_width=True)
        c2.write(f"ELA Threshold: {ela_threshold}")

        # Store the result in Firestore
        store_result_in_firestore(file_uploaded, res['label'], res['probability'])

        # Generate and upload the PDF report
        rep_gen(image_np, res['label'], res['probability'], ela_img, ela_threshold)
        st.write("Report generated and uploaded.")

def store_result_in_firestore(image, classification_result, probability):
    # Convert the image to base64 for storage in Firestore
    image_base64 = base64.b64encode(image.getvalue()).decode('utf-8')

    # Convert probability to a standard Python float
    probability = float(probability)

    # Store the data in Firestore
    data = {
        'image': image_base64,
        'classification_result': classification_result,
        'probability': probability
    }

    db.collection("image_results").add(data)
    st.write("Result stored in Firestore.")
