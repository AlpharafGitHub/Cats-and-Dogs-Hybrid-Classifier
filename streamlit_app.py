import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import requests
from io import BytesIO

# Load the pre-trained model
@st.cache(allow_output_mutation=True)
def load_model():
    model_url = "https://github.com/AlpharafGitHub/Weathering_Image_Classifier/raw/main/Weather%20Model.h5"
    response = requests.get(model_url)
    model_bytes = BytesIO(response.content)
    model = tf.keras.models.load_model(model_bytes)
    return model

# Define class labels
class_labels = ['Cloudy', 'Rain', 'Shine', 'Sunrise']

# Function to preprocess the image
def preprocess_image(image):
    img = image.resize((224, 224))  # Resize to match the model input size
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Add batch dimension
    return img_array

# Streamlit app
st.title("Weather Image Classifier")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)

    # Load the model
    model = load_model()

    # Preprocess the image
    img_array = preprocess_image(image)

    # Make predictions
    predictions = model.predict(img_array)
    predicted_class = class_labels[np.argmax(predictions)]
    st.write(f"Prediction: {predicted_class}")
