import streamlit as st
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Define image dimensions
img_height, img_width = 150, 150

# Load the trained model
model = load_model('tulu_character_recognition_model2.h5')

# Define the class indices (these should match your trained model)
# Example: class_indices = {'class1': 0, 'class2': 1, ...}
class_indices = {
    0: 'Kannada_folder1',
    1: 'Kannada_folder2',
    2: 'Kannada_folder3',
    # Add all the folder mappings based on your trained model
}

# Function to load and preprocess an image
def load_and_preprocess_image(uploaded_file):
    img = Image.open(uploaded_file).convert('RGB')
    img = img.resize((img_width, img_height))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array.astype('float32') / 255.0
    return img_array

# Streamlit App
st.title("Tulu Handwritten Character Translator")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)

    # Preprocess and predict
    img_array = load_and_preprocess_image(uploaded_file)
    predictions = model.predict(img_array)

    # Get predicted class
    predicted_class = np.argmax(predictions)
    predicted_folder = class_indices.get(predicted_class, "Unknown")

    # Display the result
    st.write(f"The image belongs to folder: {predicted_folder}")
