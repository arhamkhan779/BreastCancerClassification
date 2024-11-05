import os
import io
from flask import Flask, request, jsonify, render_template, send_from_directory
from werkzeug.utils import secure_filename
from PIL import Image
import numpy as np
import tensorflow as tf

app = Flask(__name__)

# Set up a folder for uploaded files (ensure this folder exists or create it)
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load your model (make sure the model is in the correct path)
MODEL_PATH = 'path/to/your/model.h5'
model = tf.keras.models.load_model(MODEL_PATH)

# Class labels
class_labels = {0: 'Benign', 1: 'Malignant', 2: 'Normal'}

# Function to check allowed file extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Function to preprocess the image and make predictions
def preprocess_and_predict(image):
    # Resize image to the shape your model expects (1, 224, 224, 3)
    img = image.resize((224, 224))
    img_array = np.array(img)
    
    # Check if the image has 3 channels (RGB), and if not, convert it to RGB
    if img_array.shape[-1] != 3:
        img_array = np.stack([img_array] * 3, axis=-1)  # Convert to RGB if it's grayscale

    # Normalize the image to [0, 1] by dividing by 255
    img_array = img_array / 255.0
    
    # Add batch dimension (1, 224, 224, 3)
    img_array = np.expand_dims(img_array, axis=0)

    # Make the prediction
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)[0]
    
    return class_labels[predicted_class]

@app.route('/')
def index():
    return render_template('index.html')  # Render your HTML template for the frontend

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

        # Save the file temporarily
        file.save(file_path)

        # Open the image
        image = Image.open(file_path)

        # Preprocess the image and make predictions
        try:
            prediction = preprocess_and_predict(image)
            # Return the prediction and the image URL
            image_url = f"/uploads/{filename}"
            return jsonify({'prediction': prediction, 'image_url': image_url})
        except Exception as e:
            return jsonify({'error': f'Error during prediction: {str(e)}'}), 500
    else:
        return jsonify({'error': 'Invalid file format'}), 400

# Route to serve the uploaded image
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    # Ensure the upload folder exists
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)

    # Run the Flask app
    app.run(debug=True)
