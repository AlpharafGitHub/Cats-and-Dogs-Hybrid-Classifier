import streamlit as st
import tensorflow as tf
from tensorflow.keras import models
from PIL import Image
import numpy as np
import requests
import os

# Function to download the model file
def download_model_file(url, file_path):
    response = requests.get(url)
    if response.status_code == 200:
        with open(file_path, 'wb') as f:
            f.write(response.content)
        return True
    else:
        return False

# Download the model file
model_url = "https://github.com/AlpharafGitHub/Cats-and-Dogs-Hybrid-Classifier/raw/main/cats_and_dogs_small_333.h5"
model_path = "cats_and_dogs_small_333.h5"
if not os.path.exists(model_path):
    st.write("Downloading model file...")
    if download_model_file(model_url, model_path):
        st.write("Model file downloaded successfully.")
    else:
        st.write("Failed to download model file. Please check the URL.")

# Load the model
model = tf.keras.models.load_model(model_path)

# Function to preprocess the image (same as before)
def preprocess_image(image):
    img = image.resize((28, 28))  
    img = np.array(img) / 255.0  
    img = np.expand_dims(img, axis=0)  
    return img

def main():
    st.title("Cats and Dogs Classifier")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image.', use_column_width=True)

        img_array = preprocess_image(image)

        prediction = model.predict(img_array)
        if prediction[0][0] > 0.5:
            st.write("Prediction: Dog")
        else:
            st.write("Prediction: Cat")

if __name__ == "__main__":
    main()
