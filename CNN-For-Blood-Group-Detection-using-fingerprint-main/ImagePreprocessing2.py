import cv2
import numpy as np
from PIL import Image, ImageEnhance
import os

# List of image file paths
image_paths = [r"File Path"]  # Add more paths as needed

# Define crop dimensions (width and height)
crop_width, crop_height = 100, 100  # Adjust as needed for your fingerprint images

# Directory to save processed images
output_dir = "processed_images"
os.makedirs(output_dir, exist_ok=True)

# Loop over each image path
for i, image_path in enumerate(image_paths):
    # Load the image
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"Error loading image: {image_path}")
        continue

    # Convert to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Calculate the center of the image
    center_x, center_y = gray_image.shape[1] // 2, gray_image.shape[0] // 2

    # Define the coordinates for central cropping
    x1 = max(center_x - crop_width // 2, 0)
    y1 = max(center_y - crop_height // 2, 0)
    x2 = min(center_x + crop_width // 2, gray_image.shape[1])
    y2 = min(center_y + crop_height // 2, gray_image.shape[0])

    # Crop the image around the center
    cropped_image = gray_image[y1:y2, x1:x2]

    # Apply Gaussian Blur with a smaller kernel (3x3) to reduce noise without losing fine detail
    blurred_image = cv2.GaussianBlur(cropped_image, (3, 3), 0)

    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) for fine contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_image = clahe.apply(blurred_image)

    # Resize to a larger size, using bicubic interpolation for better quality
    upscale_size = (192, 206)  # Higher resolution for improved quality
    resized_image = cv2.resize(enhanced_image, upscale_size, interpolation=cv2.INTER_CUBIC)

    # Convert to a PIL image and apply a sharpening filter
    pil_image = Image.fromarray(resized_image)
    enhancer = ImageEnhance.Sharpness(pil_image)
    sharpened_image = enhancer.enhance(1.5)  # Increase sharpness slightly

    # Save as BMP with a unique filename
    output_path = os.path.join(output_dir, f"processed_image_{i+2}.bmp")
    sharpened_image.save(output_path, format="BMP")
    
    print(f"Processed image saved at: {output_path}")
