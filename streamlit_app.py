import streamlit as st
import tensorflow as tf
import numpy as np
from io import BytesIO
from PIL import Image
import requests

# Class mapping
class_mapping = {
    0: 'cats',
    1: 'dogs',
}

# Function to load the model
@st.cache(allow_output_mutation=True)
def load_model():
    # URL for the model file on GitHub
    # model_url = "https://github.com/m3mentomor1/Breast-Cancer-Image-Classification/raw/main/CatsAndDogs_Big.h5"

    # Download the model file
    # response = requests.get(model_url)
    # model_bytes = response.content

    # Create an in-memory HDF5 file
    model = tf.keras.models.load_model("CatsAndDogs_Big.h5")

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
