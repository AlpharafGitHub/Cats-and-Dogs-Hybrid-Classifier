import streamlit as st
import tensorflow as tf
import numpy as np
from io import BytesIO
from PIL import Image
import requests
import os
import tempfile

# Class mapping
class_mapping = {
    0: 'cats',
    1: 'dogs',
}

# Function to load the model
@st.cache(allow_output_mutation=True)
def load_model():
    # URL for the model file on GitHub
    model_url = "https://github.com/AlpharafGitHub/Weathering_Image_Classifier/raw/main/Weathering Model.h5"

    # Download the model file
    response = requests.get(model_url)
    model_bytes = BytesIO(response.content)

    # Save the model bytes to a temporary file
    temp_model_path = tempfile.NamedTemporaryFile(delete=False, suffix=".h5")
    with open(temp_model_path.name, 'wb') as f:
        f.write(model_bytes.getvalue())

    # Load the model from the temporary file
    model = tf.keras.models.load_model(temp_model_path.name)

    # Clean up the temporary file
    os.unlink(temp_model_path.name)

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
