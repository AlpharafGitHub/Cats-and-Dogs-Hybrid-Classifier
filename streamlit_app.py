import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Load your model
model_path = 'https://github.com/AlpharafGitHub/Cats-and-Dogs-Hybrid-Classifier/raw/main/cats_and_dogs_small_333.h5'
model = load_model(model_path)

# Define class names
class_names = ['cats', 'dogs']

# Streamlit UI
st.title('Cats and Dogs Classifier')
st.write('Upload an image of a cat or dog, and the model will predict the class.')

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    st.image(uploaded_file, caption='Uploaded Image.', use_column_width=True)
    
    # Load and preprocess the image
    img = Image.open(uploaded_file)
    img = img.resize((150, 150))  # Resize to the same size as training images
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Predict the class
    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    predicted_class = class_names[np.argmax(score)]
    confidence = 100 * np.max(score)

    # Display the prediction
    st.write(f'Prediction: {predicted_class}')
    st.write(f'Confidence: {confidence:.2f}%')
