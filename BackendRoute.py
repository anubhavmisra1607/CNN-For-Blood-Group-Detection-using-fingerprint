from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image, ImageEnhance
import os

app = Flask(__name__)
CORS(app)

# Load the model
model = tf.keras.models.load_model('savedModel.keras')

# Define class labels
blood_group_labels = ['A+', 'A-', 'B+', 'B-', 'AB+', 'AB-', 'O+', 'O-']

# Preprocessing function
def preprocess_image(image_path):
    """
    Preprocess the image for model prediction.

    Parameters:
        image_path (str): Path to the input image.

    Returns:
        np.ndarray: Preprocessed image array ready for model input.
    """
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Error loading image: {image_path}")
    
    # Convert to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Define crop dimensions
    crop_width, crop_height = 1000, 1000  # Adjust for fingerprint images

    # Calculate the center of the image
    center_x, center_y = gray_image.shape[1] // 2, gray_image.shape[0] // 2

    # Define cropping coordinates
    x1 = max(center_x - crop_width // 2, 0)
    y1 = max(center_y - crop_height // 2, 0)
    x2 = min(center_x + crop_width // 2, gray_image.shape[1])
    y2 = min(center_y + crop_height // 2, gray_image.shape[0])

    # Crop the image
    cropped_image = gray_image[y1:y2, x1:x2]

    # Enhance contrast using CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_image = clahe.apply(cropped_image)

    # Apply Canny edge detection
    edges = cv2.Canny(enhanced_image, 50, 150)

    # Sharpen the image by combining edges with enhanced image
    sharpened_image = cv2.addWeighted(enhanced_image, 0.8, edges, 0.2, 0)

    # Upscale the image
    upscale_size = (192, 206)
    upscaled_image = cv2.resize(sharpened_image, upscale_size, interpolation=cv2.INTER_CUBIC)

    # Convert to PIL image and apply sharpening
    pil_image = Image.fromarray(upscaled_image)
    enhancer = ImageEnhance.Sharpness(pil_image)
    final_image = enhancer.enhance(1.2)

    # Resize to model input size (96x103)
    final_resized_image = final_image.resize((96, 103), Image.BICUBIC)

    # Convert back to NumPy array
    img_array = np.array(final_resized_image)

    # Normalize image and add batch dimension
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=-1)  # Add channel dimension if grayscale
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

        # Clean up the temporary file
        os.remove(temp_path)

        return jsonify({'blood_group': blood_group})
    except Exception as e:
        return jsonify({'error': f'An error occurred: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True)
