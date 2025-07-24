from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
import cv2
import os

app = Flask(__name__)
CORS(app)

# Load the model
model = tf.keras.models.load_model(r'C:\Users\USER\Desktop\Stuff\Code\BloodGroupDetection\savedModel.keras')

# Define class labels
blood_group_labels = ['A+', 'A-', 'AB+', 'AB-', 'B+', 'B-', 'O+', 'O-']

# Minimum preprocessing function
def preprocess_image(image_path):
    """
    Minimal preprocessing: Resize and normalize the image.

    Parameters:
        image_path (str): Path to the input image.

    Returns:
        np.ndarray: Preprocessed image array ready for model input.
    """
    # Load the image (any color format supported by OpenCV)
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Error loading image: {image_path}")

    # Resize the image to the model input size
    resized_image = cv2.resize(image, (96, 103))

    # Normalize pixel values to [0, 1]
    img_array = resized_image / 255.0

    # Expand dimensions to match model input
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

@app.route('/predict', methods=['POST'])
def predict_blood_group():
    """
    Route to preprocess image and predict blood group.
    """
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        # Save the uploaded file temporarily
        temp_path = os.path.join('uploads', file.filename)
        os.makedirs('uploads', exist_ok=True)
        file.save(temp_path)

        # Preprocess the image
        img_array = preprocess_image(temp_path)

        # Predict blood group
        predictions = model.predict(img_array)
        predicted_class_index = np.argmax(predictions, axis=1)[0]
        blood_group = blood_group_labels[predicted_class_index]
        
        print("Predicted blood group ", blood_group)

        # Clean up the temporary file
        os.remove(temp_path)

        return jsonify({'blood_group': blood_group})
    except Exception as e:
        return jsonify({'error': f'An error occurred: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True)
