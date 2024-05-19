import streamlit as st
import tensorflow as tf
import numpy as np
from io import BytesIO
from PIL import Image
import requests
import os

# Class mapping
class_mapping = {
    0: 'cats',
    1: 'dogs',
}

# Load the pre-trained model
@st.cache(allow_output_mutation=True)
def load_model():
    model_url = "https://github.com/AlpharafGitHub/Weathering_Image_Classifier/raw/main/Weather_Model.h5"
    model_path = tf.keras.utils.get_file("weather_model.h5", origin=model_url, cache_subdir=os.path.abspath("."))
    model = tf.keras.models.load_model(model_path)
    return model

# Function to preprocess and make predictions
def predict(image, model):
    # Preprocess the image
    image_array = image.resize((224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(image)
    img_array = tf.expand_dims(img_array, 0)

    # Make prediction
    predictions = model.predict(img_array)

    # Get the predicted class
    predicted_class = class_mapping[np.argmax(predictions[0])]
    return predicted_class

# Streamlit app
st.title("Weather Image Classifier")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)

    # Load the model
    model = load_model()

    # Make predictions
    predicted_class = predict(image, model)
    st.write(f"Prediction: {predicted_class}")
