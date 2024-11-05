import os
import numpy as np
import tensorflow as tf
from PIL import Image
import streamlit as st

# Set up paths for the model and the upload folder
UPLOAD_FOLDER = 'uploads'
MODEL_PATH = 'artifacts/training/model.h5'  # Update path if necessary

# Load the trained model
model = tf.keras.models.load_model(MODEL_PATH)

# Class labels for prediction
class_labels = {0: 'Benign', 1: 'Malignant', 2: 'Normal'}

# Function to preprocess image and make predictions
def preprocess_and_predict(image):
    # Ensure the image is in RGB format, even if it has an alpha channel (RGBA)
    if image.mode != 'RGB':
        image = image.convert('RGB')

    # Resize image to 224x224 as expected by the model
    img = image.resize((224, 224))
    img_array = np.array(img)

    # Ensure the image has 3 channels (RGB)
    if img_array.shape[-1] != 3:
        img_array = np.stack([img_array] * 3, axis=-1)  # Convert to RGB if grayscale

    # Normalize image to [0, 1]
    img_array = img_array / 255.0

    # Add batch dimension (1, 224, 224, 3)
    img_array = np.expand_dims(img_array, axis=0)

    # Make the prediction
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)[0]

    return class_labels[predicted_class]

# Streamlit application
def main():
    # Custom CSS for styling
    st.markdown("""
        <style>
            /* General Styles */
            body {
                font-family: 'Arial', sans-serif;
                margin: 0;
                padding: 0;
                background-image: url('//pp.jpg');  /* Use relative path to image */
                background-size: cover;
                background-position: center center;
                background-attachment: fixed;
                color: #fff;
                height: 100vh;
                display: flex;
                justify-content: center;
                align-items: center;
                text-align: center;
            }

            .container {
                position: relative;
                max-width: 800px;
                width: 100%;
                padding: 40px;
                background: rgba(255, 255, 255, 0.8);
                border-radius: 15px;
                box-shadow: 0 10px 20px rgba(0, 0, 0, 0.2);
            }

            h1 {
                font-size: 36px;
                margin-bottom: 20px;
                color: #f04e84;
            }

            .description {
                font-size: 18px;
                margin-bottom: 20px;
                color: #ff66b2;
            }

            .upload-section {
                margin-top: 30px;
                padding: 30px;
                background: #fff;
                border-radius: 10px;
                box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.1);
            }

            .drop-zone {
                border: 3px dashed #f04e84;
                border-radius: 10px;
                padding: 40px;
                cursor: pointer;
                transition: all 0.3s ease;
                background: rgba(255, 192, 203, 0.3);
            }

            .drop-zone:hover {
                background: rgba(255, 192, 203, 0.5);
                border-color: #f04e84;
            }

            .prediction {
                margin-top: 30px;
                padding: 15px;
                font-size: 20px;
                background: #fff;
                border-radius: 10px;
                box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.1);
            }

            .prediction p {
                font-weight: bold;
                color: #333;
            }

            .result-text {
                font-size: 24px;
                color: #f04e84;
                margin-top: 10px;
            }

            .error-text {
                font-size: 24px;
                color: #ff4d4d;
                margin-top: 10px;
            }

            .uploaded-image {
                max-width: 100%;
                max-height: 300px;
                margin-top: 20px;
                border-radius: 10px;
                box-shadow: 0 10px 15px rgba(0, 0, 0, 0.1);
            }

            /* Responsive Design */
            @media (max-width: 768px) {
                .container {
                    padding: 20px;
                    margin: 10px;
                }

                .drop-zone {
                    padding: 20px;
                }

                .prediction {
                    font-size: 18px;
                }
            }
        </style>
    """, unsafe_allow_html=True)

    # Title
    st.title("Breast Cancer Classification")
    st.markdown("Upload an image to predict whether it is **Benign**, **Malignant**, or **Normal**.")

    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        # Open the uploaded image using PIL
        image = Image.open(uploaded_file)

        # Display the uploaded image
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Prediction
        prediction = preprocess_and_predict(image)

        # Show prediction result
        st.subheader(f"Prediction: {prediction}")

if __name__ == "__main__":
    main()
