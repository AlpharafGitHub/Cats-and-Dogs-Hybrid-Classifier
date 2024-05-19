import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import requests
import os

# Class mapping
class_mapping = {
    0: 'cats',
    1: 'dogs',
}

# Function to load the model
@st.cache(allow_output_mutation=True)
def load_model():
    # URL for the model file on GitHub
    model_url = "https://github.com/AlpharafGitHub/Cats-and-Dogs-Hybrid-Classifier/raw/main/cats_and_dogs_small_333.h5"

    # Download the model file
    response = requests.get(model_url)
    model_path = "cats_and_dogs_small_333.h5"
    with open(model_path, "wb") as f:
        f.write(response.content)

    # Load the model
    model = tf.keras.models.load_model(model_path)

    return model

# Function to preprocess and make predictions
def predict(image, model):
    # Preprocess the image
    img_array = np.array(image)
    img_array = tf.image.resize(img_array, (256, 256))  # Adjust the size as per your model requirements
    img_array = tf.expand_dims(img_array, 0)  # Add batch dimension
    img_array = img_array / 255.0  # Normalize

    # Make prediction
    predictions = model.predict(img_array)

    # Get the predicted class
    predicted_class = class_mapping[np.argmax(predictions[0])]
    return predicted_class

# Streamlit app
st.title("Cats and Dogs Classifier")
uploaded_file = st.file_uploader("Choose a image of a cat or dog...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)

    # Load the model
    model = load_model()

    # Make predictions
    predicted_class = predict(image, model)
    st.write(f"Prediction: {predicted_class}")
