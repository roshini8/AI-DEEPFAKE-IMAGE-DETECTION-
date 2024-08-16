# Your Streamlit app script
import streamlit as st
import torch
import torch.nn.functional as F
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
import cv2
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
import numpy as np
import warnings
import jinja2
import pdfkit
import os
import base64
import firebase_admin
from firebase_admin import credentials, firestore, storage  # Import the storage module
from datetime import datetime
import pandas as pd
import matplotlib as plt
from sklearn.ensemble import IsolationForest
import seaborn as sns

warnings.filterwarnings("ignore")

# Check if GPU is available, otherwise use CPU
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

mtcnn = MTCNN(
    select_largest=False,
    post_process=False,
    device=DEVICE
).eval()

# Load face authentication model
model = InceptionResnetV1(
    pretrained="vggface2",
    classify=True,
    num_classes=1,
    device=DEVICE
)

checkpoint = torch.load("Streamlit/resnetinceptionv1_epoch_32.pth", map_location=torch.device('cpu'))
model.load_state_dict(checkpoint['model_state_dict'])
model.to(DEVICE)
model.eval()

# Initialize Firebase
if not firebase_admin._apps:
    cred = credentials.Certificate("Streamlit/key.json")
    firebase_admin.initialize_app(cred, {'storageBucket': 'myapplication-71465.appspot.com'})

# Initialize Firestore client
db = firestore.client()


def rep_gen(face_with_mask, prediction, confidences, uploaded_file):
    # Encode face_with_mask and uploaded_file to Base64
    encoded_face_with_mask = cv2.imencode('.jpg', face_with_mask)[1].tobytes()
    face_with_mask_base64 = base64.b64encode(encoded_face_with_mask).decode('utf-8')

    # Encode uploaded_file to Base64
    uploaded_file.seek(0)  # Reset file pointer
    uploaded_file_base64 = base64.b64encode(uploaded_file.read()).decode('utf-8')

    context = {
        'prediction': prediction,
        'confidences': confidences,
        'uploaded_file': uploaded_file_base64,
        'face_with_mask': face_with_mask_base64,
    }

    template_loader = jinja2.FileSystemLoader('./')
    template_env = jinja2.Environment(loader=template_loader)
    template = template_env.get_template("Streamlit/image_classification_report.html")
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


def upload_image_to_storage(image_path, storage_path):
    """Upload the image to Firebase Storage"""
    bucket = storage.bucket('myapplication-71465.appspot.com')  # Specify your storage bucket name here
    blob = bucket.blob(storage_path)
    blob.upload_from_filename(image_path)


def predict(input_image):
    """Predict the label of the input_image"""
    face = mtcnn(input_image)
    if face is None:
        raise Exception('No face detected')

    # Process the face for face authentication
    face = face.unsqueeze(0)  # add the batch dimension
    face = F.interpolate(face, size=(256, 256), mode='bilinear', align_corners=False)

    prev_face = face.squeeze(0).permute(1, 2, 0).cpu().detach().int().numpy()
    prev_face = prev_face.astype('uint8')

    face = face.to(DEVICE)
    face = face.to(torch.float32)
    face = face / 255.0
    face_image_to_plot = face.squeeze(0).permute(1, 2, 0).cpu().detach().int().numpy()

    target_layers = [model.block8.branch1[-1]]
    use_cuda = True if torch.cuda.is_available() else False
    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=use_cuda)
    targets = [ClassifierOutputTarget(0)]

    grayscale_cam = cam(input_tensor=face, targets=targets, eigen_smooth=True)
    grayscale_cam = grayscale_cam[0, :]
    visualization = show_cam_on_image(face_image_to_plot, grayscale_cam, use_rgb=True)
    face_with_mask = cv2.addWeighted(prev_face, 1, visualization, 0.5, 0)

    # Adjust the threshold as needed
    threshold = 0.5

    with torch.no_grad():
        output = torch.sigmoid(model(face).squeeze(0))
        confidence_score = output.item()

        if confidence_score <= threshold:
            prediction = "fake"
        else:
            prediction = "real"

        confidences = {
            'real': confidence_score,  # Confidence for "real" class
            'fake': 1 -confidence_score  # Confidence for "fake" class
        }

    return prediction, confidences, face_with_mask


def app():
    if st.session_state.get("my_input") == "true":

        st.title("Face Forgery Detection")

        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image.", use_column_width=True)
            st.write("")
            st.write("Processing...")

            # Perform face authentication
            prediction, confidences, face_with_mask = predict(image)


            st.image(face_with_mask, caption="Face with Explainability", use_column_width=True)
            # Display face authentication results



            st.write(f"Face Authentication Prediction: {prediction}")
            st.write(f"Confidence - Real: {confidences['real']:.4f}, Fake: {confidences['fake']:.4f}")
            # Generate and display metadata analysis
            image_format = image.format
            image_mode = image.mode
            image_size = image.size

            st.subheader("Metadata Analysis")
            st.write(f"Image Format: {image_format}")
            st.write(f"Image Mode: {image_mode}")
            st.write(f"Image Size: {image_size}")

            if st.button("Download"):
                # Generate a unique filename for the temporary image
                unique_filename = f"temp_image_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                image_path = os.path.join('./', unique_filename)
                image.save(image_path)

                rep_gen(face_with_mask, prediction, confidences, uploaded_file)

                # Upload image to Firebase Storage
                storage_path = f"images/{unique_filename}"
                upload_image_to_storage(image_path, storage_path)

                # Store the result in Firestore
                store_result_in_firestore(storage_path, prediction, confidences)

                # Remove the temporary image file
                os.remove(image_path)

    else:
        st.write("Please log in to access the main page.")


def store_result_in_firestore(storage_path, prediction, confidences):
    # Get the username from session_state
    username = st.session_state.username

    if username is not None:
        # Get the user profile document reference based on the username
        user_profile_ref = db.collection("userprofile").document(username)

        # Create a subcollection called "image_results" for the user profile
        image_results_ref = user_profile_ref.collection("image_results")

        # Store the data in the "image_results" subcollection
        data = {
            'Image': storage_path,  # Store the storage path instead of base64 image data
            'Prediction': prediction,
            'Confidences_real': confidences['real'],
            'Confidences_fake': confidences['fake'],
        }

        image_results_ref.add(data)
        st.write("Result stored in Firestore.")

