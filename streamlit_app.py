import streamlit as st
import tensorflow as tf
from tensorflow.keras import models
from PIL import Image
import numpy as np

# Load the model
model_path = "https://github.com/AlpharafGitHub/Cats-and-Dogs-Hybrid-Classifier/raw/main/cats_and_dogs_small_333.h5"
model = tf.keras.models.load_model(model_path)

# Function to preprocess the image
def preprocess_image(image):
    img = image.resize((28, 28))  # Resize image to match model's expected sizing
    img = np.array(img) / 255.0  # Normalize pixel values to range [0, 1]
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

def main():
    st.title("Cats and Dogs Classifier")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image.', use_column_width=True)

        # Preprocess the image
        img_array = preprocess_image(image)

        # Make prediction
        prediction = model.predict(img_array)
        if prediction[0][0] > 0.5:
            st.write("Prediction: Dog")
        else:
            st.write("Prediction: Cat")

if __name__ == "__main__":
    main()
