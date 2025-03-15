import cv2
import numpy as np

def find_repetitive_patterns(image_path):
    # Load the image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Apply Gaussian Blur to reduce noise
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    
    # Initialize ORB detector
    orb = cv2.ORB_create()
    
    # Detect keypoints and descriptors
    keypoints, descriptors = orb.detectAndCompute(blurred, None)
    
    # Use BFMatcher to find matches
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(descriptors, descriptors)
    
    # Sort matches by distance
    matches = sorted(matches, key=lambda x: x.distance)
    
    # Draw matches on the image
    result = cv2.drawMatches(image, keypoints, image, keypoints, matches[:10], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    
    # Display the result
    cv2.imshow('Repetitive Patterns', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Example usage
find_repetitive_patterns('path/to/image.jpg')