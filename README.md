# AksharaSetu: The Character Bridge Between Tulu and Kannada  
Tulu, a Dravidian language, has recently been assigned a Unicode standard but lacks system-wide support. This project aims to develop a Tulu OCR system that recognizes handwritten Tulu characters and translates them into their corresponding Kannada characters.

### The project includes:
- Handwritten character recognition using a trained machine learning model.
- Image preprocessing, including binarization, to enhance the quality of the input.
- A GUI interface (Tkinter) for users to upload images of handwritten Tulu characters for translation.
- A prediction system that outputs Kannada equivalents of Tulu characters.

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Requirements](#requirements)
- [Usage](#usage)
- [Preprocessing Steps](#preprocessing-steps)
- [Model Details](#model-details)
- [GUI Interface](#gui-interface)
- [Results](#results)
- [Future Improvements](#future-improvements)
## Introduction
Tulu, a Dravidian language, has recently been assigned a Unicode standard but lacks system-wide support. This project aims to develop a Tulu OCR system that recognizes handwritten Tulu characters and translates them into their corresponding Kannada characters.

### The project includes:

- Handwritten character recognition using a trained machine learning model.
- Image preprocessing, including binarization, to enhance the quality of the input.
- A GUI interface (Tkinter) for users to upload images of handwritten Tulu characters for translation.
- A prediction system that outputs Kannada equivalents of Tulu characters.
## Features
- Handwritten Tulu character recognition.
- Translation into Kannada characters.
- Image preprocessing (grayscale conversion, binarization).
- Simple and user-friendly GUI using Tkinter.
- Dataset creation and training for the recognition of Tulu characters.
- Two-character prediction and mapping using an Excel sheet for validation.
- Image upload and selection for testing
## Requirements
To run this project, you will need the following:

- Python 3.8 or later
- Tkinter (for GUI)
- OpenCV
- NumPy
- TensorFlow/Keras (for model training and prediction)
- Matplotlib (optional, for image visualization)

## Usage
- Start the application by running main.py.
- Upload an image of a handwritten Tulu character.
- The system will preprocess the image (grayscale conversion, binarization) and then predict the corresponding Kannada character.
- You can view the results in the GUI, with the original image and translated Kannada character displayed.

## Preprocessing Steps
Before recognizing the handwritten characters, the images undergo a series of preprocessing steps to improve the quality of predictions. These steps include:

- Grayscale Conversion: The input images are first converted to grayscale, removing any color information, which simplifies the image for further processing.

- Binarization: The grayscale image is binarized, converting it into a black-and-white image where the character stands out clearly against the background. This helps in distinguishing the character from noise and background.

Example of an image before and after binarization:

Original Image	           
![DATASET_0001](https://github.com/user-attachments/assets/a008ac5e-772a-45bf-8014-05d8add3d890)

Binarized Image

![1](https://github.com/user-attachments/assets/4dc449d1-b742-42ba-a17d-76440b11c027)

- Augmentation: To enhance the robustness of the model and increase the dataset size, various augmentation techniques were applied, such as:

  - Rotation: Images are rotated slightly to simulate different handwriting angles.
  - Scaling: Characters are scaled to introduce variations in size.
  - Shifting: The characters are shifted slightly to mimic natural variance in positioning.
  - Background Addition: A black background is applied uniformly to all images, which helps in contrast enhancement and model consistency.
    
    ![ಅ_1](https://github.com/user-attachments/assets/4a33a2fd-8521-48ef-81d6-5a3a58352895)

- Noise Reduction: Any noise in the background is removed to make the characters more prominent.

- Resizing: The images are resized to a uniform size for consistent input to the model.


## Model Details
The model is built using a sequential neural network architecture in TensorFlow/Keras. It uses convolutional layers followed by max pooling to extract features from the input images. The model is trained on a dataset of Tulu characters labeled with corresponding Kannada translations.

#### Input: Preprocessed images of handwritten Tulu characters (binarized and resized).
#### Output: Predicted Kannada character.
### Training
The model was trained on a custom dataset of Tulu characters, labeled with Kannada folder names corresponding to their translations.
## Dataset
The dataset includes images of handwritten Tulu characters, each labeled with its corresponding Kannada translation. Folders are named using Kannada characters, and the images are saved within these folders. The dataset can be expanded with more samples to improve accuracy.
#### Input Image: Handwritten character images with a black background.
#### Labels: Folder names in Kannada characters.
## GUI Interface
The graphical interface built using Tkinter allows users to:
- Upload images for recognition.
- View predictions (Kannada character).
- Navigate between images within a set.
## Results
The model has achieved good accuracy in recognizing Tulu characters, and the dual prediction system helps improve reliability. It displays the predicted Kannada character side by side with the input image for better understanding.
![image](https://github.com/user-attachments/assets/64a32032-d20a-430b-9a06-a9cde8ded8ec)
## Future Improvements
Enhance dataset size for better accuracy.
Implement support for multiple Tulu word recognition.
Add options to handle noise and improve character segmentation.
Further optimize the user interface for better usability.
Deploy the system as a web-based application for broader access
