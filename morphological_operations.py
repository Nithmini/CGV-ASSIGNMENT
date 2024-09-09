import cv2
import numpy as np


def apply_dilation(image, kernel_size=(3, 3), iterations=1):
    """Apply dilation to the image."""
    kernel = np.ones(kernel_size, np.uint8)
    return cv2.dilate(image, kernel, iterations=iterations)


def apply_erosion(image, kernel_size=(3, 3), iterations=1):
    """Apply erosion to the image."""
    kernel = np.ones(kernel_size, np.uint8)
    return cv2.erode(image, kernel, iterations=iterations)