import cv2
import numpy as np


def apply_adaptive_threshold(image):
    """Apply adaptive thresholding to better capture text and edges."""
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply adaptive thresholding
    adaptive_thresh = cv2.adaptiveThreshold(
        gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

    return adaptive_thresh


def find_receipt_contour(image, binary_image):
    """Find the largest contour which should represent the receipt boundary."""
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Initialize variables for the largest rectangle contour
    largest_rectangle = None
    largest_area = 0

    for contour in contours:
        area = cv2.contourArea(contour)

        # Filter based on area size; we are interested in large contours
        if area > 600:
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)

            # Check if the contour is a rectangle
            if len(approx) > 2:
                if area > largest_area:
                    largest_rectangle = approx
                    largest_area = area

    # Draw the largest rectangle on the original image if found
    image_with_contours = image.copy()
    if largest_rectangle is not None:
        cv2.drawContours(image_with_contours, [largest_rectangle], -1, (0, 255, 0), 3)

    return image_with_contours


if __name__ == "__main__":
    image_path = 'images/Recept-I.png'  # Use the correct image path
    image = cv2.imread(image_path)

    # Step 1: Apply adaptive thresholding
    binary_image = apply_adaptive_threshold(image)

    # Step 2: Find the receipt contour by filtering and approximation
    image_with_contours = find_receipt_contour(image, binary_image)

    # Display the results
    cv2.imshow('Binary Image', binary_image)
    cv2.imshow('Image with Contours', image_with_contours)
    cv2.waitKey(0)
    cv2.destroyAllWindows()