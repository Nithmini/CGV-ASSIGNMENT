# sharpening.py
import cv2
import numpy as np


def apply_sharpening(image):
    """Apply sharpening to the image."""
    kernel = np.array([[-1, -1, -1],
                       [-1, 9, -1],
                       [-1, -1, -1]])
    return cv2.filter2D(image, -1, kernel)