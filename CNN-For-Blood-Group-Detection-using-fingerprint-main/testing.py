import cv2
import numpy as np
from PIL import Image, ImageEnhance
import os
import tensorflow as tf

def preprocess_and_save_image(image_path, output_dir, model):
    """
    Preprocess the image for prediction, save the processed image, and predict blood group.
    
    Parameters:
        image_path (str): Path to the input image.
        output_dir (str): Directory where processed images will be saved.
        model (tensorflow.keras.Model): The trained model for prediction.
    
    Returns:
        str: Predicted blood group.
    """
    # Load the image (keeping the RGB data)
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error loading image: {image_path}")
        return "Error loading image"

    # Convert the image to RGB (if it's in BGR)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Define crop dimensions (adjust as needed)
    crop_width, crop_height = 100, 100  # Adjust as needed for your fingerprint images

    # Calculate the center of the image
    center_x, center_y = image_rgb.shape[1] // 2, image_rgb.shape[0] // 2

    # Define cropping coordinates
    x1 = max(center_x - crop_width // 2, 0)
    y1 = max(center_y - crop_height // 2, 0)
    x2 = min(center_x + crop_width // 2, image_rgb.shape[1])
    y2 = min(center_y + crop_height // 2, image_rgb.shape[0])

    # Crop the image
    cropped_image = image_rgb[y1:y2, x1:x2]

    # Apply Canny edge detection (RGB edges)
    edges = cv2.Canny(cv2.cvtColor(cropped_image, cv2.COLOR_RGB2GRAY), 50, 150)

    # Convert edges to 3 channels (RGB)
    edges_rgb = cv2.merge([edges, edges, edges])

    # Sharpen the image by blending it with edges
    sharpened_image = cv2.addWeighted(cropped_image, 0.8, edges_rgb, 0.2, 0)

    # Enhance the image's contrast using CLAHE (Contrast Limited Adaptive Histogram Equalization)
    lab_image = cv2.cvtColor(sharpened_image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab_image)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab_image = cv2.merge((l, a, b))
    enhanced_image = cv2.cvtColor(lab_image, cv2.COLOR_LAB2RGB)

    # Resize the image to model input size (96x103) for prediction
    final_resized_image = cv2.resize(enhanced_image, (96, 103), interpolation=cv2.INTER_CUBIC)

    # Save the processed image in the specified directory
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "processed_image.jpg")
    Image.fromarray(final_resized_image).save(output_path, format="JPEG")
    
    # Load the image again after saving
    img_array = cv2.imread(output_path)
    img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Normalize the image

    # Predict the blood group using the model
    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    
    # Define the labels for blood group classes
    blood_group_labels = ['A+', 'A-', 'AB+', 'AB-', 'B+', 'B-', 'O+', 'O-']
    blood_group = blood_group_labels[predicted_class_index]
    
    print(f"Processed image saved at: {output_path}")
    return blood_group

# Load the model from the specified saved location
model_path = r'C:\Users\USER\Desktop\Stuff\Code\BloodGroupDetection\savedModel.keras'  # Replace with your actual model path
model = tf.keras.models.load_model(model_path)

# Example usage
image_path = r"C:\Downloads\TestingA-.BMP"  # Replace with the actual image path
output_dir = 'processed_images'  # Specify the output directory where processed images will be saved
blood_group = preprocess_and_save_image(image_path, output_dir, model)

print(f"The predicted blood group is: {blood_group}")
