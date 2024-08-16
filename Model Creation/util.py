import tensorflow as tf
from PIL import Image
import numpy as np
# Your main script


import io

model = None
labels = ['real', 'fake']

def load_model():
    global model
    model = tf.keras.models.load_model('dfd_weights.h5')
    #model = tf.keras.models.load_model('dffnetv2B0.h5')
def classify_image(file_path):
    if model is None:
        load_model()

    image = Image.open(file_path) # reading the image
    image = image.resize((128, 128)) # resizing the image to fit the trained model
    img = np.asarray(image) # converting it to numpy array
    img = np.expand_dims(img, 0)
    predictions = model.predict(img) # predicting the label
    label = labels[np.argmax(predictions[0])] # extracting the label with maximum probablity
    probab = float(round(predictions[0][np.argmax(predictions[0])]*100, 2))

    result = {
        'label': label,
        'probablity': probab
    }

    return result


