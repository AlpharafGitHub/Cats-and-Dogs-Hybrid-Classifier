import streamlit as st
import tensorflow as tf
from tensorflow.keras import models
from PIL import Image
import numpy as np
import requests
import os

# Function to download the model file (same as before)
# Load the model
model_path = "https://github.com/AlpharafGitHub/Cats-and-Dogs-Hybrid-Classifier/raw/main/cats_and_dogs_small_333.h5"
model = tf.keras.models.load_model(model_path, compile=False)

# Function to preprocess the image (same as before)
def preprocess_image(image):
    img = image.resize((150, 150))  
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
