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
    'ಅ': 1, 'ಆ': 2, 'ಇ': 3, 'ಈ': 4, 'ಉ': 5,
    'ಊ': 6, 'ಋ': 7, 'ೠ': 8, 'ಎ': 9, 'ಏ': 10,
    'ಐ': 11, 'ಒ': 12, 'ಔ': 13, 'ಅಂ': 14, 'ಅಃ': 15,
    'ಕ': 16, 'ಖ': 17, 'ಗ': 18, 'ಘ': 19, 'ಙ': 20,
    'ಚ': 21, 'ಛ': 22, 'ಜ': 23, 'ಝ': 24, 'ಞ': 25,
    'ಟ': 26, 'ಠ': 27, 'ಡ': 28, 'ಢ': 29, 'ಣ': 30,
    'ತ': 31, 'ಥ': 32, 'ದ': 33, 'ಧ': 34, 'ನ': 35,
    'ಪ': 36, 'ಫ': 37, 'ಬ': 38, 'ಭ': 39, 'ಮ': 40,
    'ಯ': 41, 'ರ': 42, 'ಲ': 43, 'ವ': 44, 'ಶ': 45,
    'ಷ': 46, 'ಸ': 47, 'ಹ': 48, 'ಳ': 49
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
