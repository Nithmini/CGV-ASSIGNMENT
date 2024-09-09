import cv2
from grayscale import convert_to_grayscale
from binarization import apply_binarization
from morphological_operations import apply_dilation, apply_erosion
from sharpening import apply_sharpening
from edge_detection import (
    apply_clahe,
    apply_adaptive_threshold,
    apply_morphology,
    find_receipt_contours,
    combine_overlapping_rectangles
)
from transformation import draw_transformed_receipts

def load_image(image_path):
    """Load the image from the specified path."""
    image = cv2.imread(image_path)
    return image

def apply_edge_detection_and_transformation(image):
    """Apply edge detection and transformation to the image."""
    # Step 1: Enhance contrast and apply adaptive thresholding
    cl1 = apply_clahe(image)
    binary_image = apply_adaptive_threshold(cl1)

    # Step 2: Apply morphological operations
    morphed_image = apply_morphology(binary_image)

    # Step 3: Detect all contours that could be receipts, with a minimum height of 200 pixels
    bounding_boxes = find_receipt_contours(morphed_image, min_height=200)

    # Step 4: Combine overlapping rectangles
    combined_boxes = combine_overlapping_rectangles(bounding_boxes)

    # Step 5: Apply the perspective transformation to each detected receipt
    transformed_receipts = draw_transformed_receipts(image, combined_boxes)

    return transformed_receipts[0]  # Return the first transformed receipt for further processing

def main(image_path, operations):
    image = load_image(image_path)

    # Apply edge detection and transformation
    image = apply_edge_detection_and_transformation(image)

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
    operations = ['grayscale', 'dilation', 'sharpening', 'binarization']
    main(image_path, operations)