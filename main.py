# main.py
import cv2
from grayscale import convert_to_grayscale
from binarization import apply_binarization
from morphological_operations import apply_dilation, apply_erosion
from sharpening import apply_sharpening


def load_image(image_path):
    """Load the image from the specified path."""
    image = cv2.imread(image_path)
    return image


def main(image_path):
    image = load_image(image_path)
    gray_image = convert_to_grayscale(image)
    binary_image = apply_binarization(gray_image)
    dilated_image = apply_dilation(binary_image)
    eroded_image = apply_erosion(dilated_image)
    sharpened_image = apply_sharpening(eroded_image)
    cv2.imshow('Final Image', sharpened_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    image_path = "images/Recept-I.png"
    main(image_path)