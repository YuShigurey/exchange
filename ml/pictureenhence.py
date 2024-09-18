import cv2
import os

def enhance_image(image_path, output_path):
    # Read the image
    image = cv2.imread(image_path)
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian Blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Enhance contrast using histogram equalization
    enhanced = cv2.equalizeHist(blurred)
    
    # Detect edges using Canny
    edges = cv2.Canny(enhanced, 100, 200)
    
    # Save the processed image
    cv2.imwrite(output_path, edges)

# Example usage
input_directory = 'path/to/input/images'
output_directory = 'path/to/output/images'

# Ensure output directory exists
os.makedirs(output_directory, exist_ok=True)

# Process all images in the input directory
for filename in os.listdir(input_directory):
    if filename.endswith('.jpg') or filename.endswith('.png'):
        input_path = os.path.join(input_directory, filename)
        output_path = os.path.join(output_directory, filename)
        enhance_image(input_path, output_path)