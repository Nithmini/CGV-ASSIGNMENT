import cv2
from grayscale import convert_to_grayscale
from binarization import apply_binarization
from morphological_operations import apply_dilation, apply_erosion
from sharpening import apply_sharpening


def load_image(image_path):
    """Load the image from the specified path."""
    image = cv2.imread(image_path)
    return image


def main(image_path, operations):
    image = load_image(image_path)

    # Dictionary of available operations
    operation_functions = {
        'grayscale': convert_to_grayscale,
        'binarization': apply_binarization,
        'dilation': apply_dilation,
        'erosion': apply_erosion,
        'sharpening': apply_sharpening
    }

    # Execute specified operations in order
    for operation in operations:
        if operation in operation_functions:
            image = operation_functions[operation](image)
        else:
            print(f"Warning: '{operation}' is not a valid operation")

    cv2.imshow('Processed Image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    image_path = "images/Recept-II.png"
    # Define the order and selection of operations
    # operations = ['grayscale', 'binarization', 'dilation', 'erosion', 'sharpening']
    operations = ['grayscale', 'dilation','sharpening', 'binarization', ]
    main(image_path, operations)