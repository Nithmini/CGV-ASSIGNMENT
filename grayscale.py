import cv2


def convert_to_grayscale(image):
    """Convert the input image to grayscale."""
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# images/Recept-I.png