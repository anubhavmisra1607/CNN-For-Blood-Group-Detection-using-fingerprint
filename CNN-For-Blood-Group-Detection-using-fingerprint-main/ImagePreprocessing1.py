import cv2
import numpy as np
from PIL import Image, ImageEnhance
import os

# List of image file paths
image_paths = [r"File Path"]  # Add more paths as needed

# Define crop dimensions (width and height)
crop_width, crop_height = 1000, 1000  # Adjust as needed for your fingerprint images

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

    # Apply CLAHE for contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_image = clahe.apply(cropped_image)

    # Apply Canny edge detection
    edges = cv2.Canny(enhanced_image, 50, 150)

    # Combine edges with enhanced image using a weighted sum for better sharpness
    sharpened_image = cv2.addWeighted(enhanced_image, 0.8, edges, 0.2, 0)

    # Upscale image to a larger size with bicubic interpolation for improved resolution
    upscale_size = (192, 206)
    upscaled_image = cv2.resize(sharpened_image, upscale_size, interpolation=cv2.INTER_CUBIC)

    # Convert to a PIL image and apply a slight sharpening filter
    pil_image = Image.fromarray(upscaled_image)
    enhancer = ImageEnhance.Sharpness(pil_image)
    final_image = enhancer.enhance(1.2)  # Adjust the sharpness factor as needed

    # Resize to target resolution (96x103) for output while retaining quality
    final_resized_image = final_image.resize((96, 103), Image.BICUBIC)

    # Save as BMP with a unique filename
    output_path = os.path.join(output_dir, f"processed_image_{i+1}.bmp")
    final_resized_image.save(output_path, format="BMP")
    
    print(f"Processed image saved at: {output_path}")
