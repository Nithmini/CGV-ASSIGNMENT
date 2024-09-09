import cv2

# Load the Grayscale Image
import cv2


def apply_binarization(gray_image, threshold_value=150):
    """Apply binarization (thresholding) to the grayscale image."""
    _, binary_image = cv2.threshold(gray_image, threshold_value, 255, cv2.THRESH_BINARY_INV)
    return binary_image