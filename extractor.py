import cv2
import numpy as np

def extract_brain_contour(img):
    """
    Extract brain region contour and crop the MRI image.
    """
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian Blur
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply thresholding to segment
    _, thresh = cv2.threshold(blur, 45, 255, cv2.THRESH_BINARY)

    # Fill holes
    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=2)

    # Find contours
    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        return img  # fallback if no contour found

    # Find the largest contour (brain)
    c = max(contours, key=cv2.contourArea)

    # Get bounding box and crop
    x, y, w, h = cv2.boundingRect(c)
    cropped_img = img[y:y+h, x:x+w]

    return cropped_img
