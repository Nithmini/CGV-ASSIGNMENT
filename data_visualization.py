import cv2
import re
import matplotlib.pyplot as plt
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
from ocr_functions import extract_text_from_image
from format_output import print_formatted_text


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


def get_operations_for_image(image_path):
    """Return the list of operations based on the image path."""
    # Define the operations for specific image paths or patterns
    operations_by_image = {
        "images/Recept-I.png": ['grayscale', 'dilation', 'sharpening', 'erosion'],
        "images/Recept-II.png": ['grayscale', 'sharpening', 'binarization'],
        "images/Recept-III.png": ['grayscale', 'sharpening'],
        "images/Recept-IV.png": ['grayscale', 'sharpening'],
        "images/Recepts.png": ['grayscale', 'sharpening', 'binarization'],
    }

    # Default operations if the image path doesn't match any specific case
    default_operations = ['grayscale', 'binarization']

    # Return the operations based on the image path, or the default if not found
    return operations_by_image.get(image_path, default_operations)


def extract_total_or_subtotal(extracted_text):
    """Extract the total or subtotal from the formatted text."""
    # Improved regex to match 'Sub Total', 'Total', or similar keywords with various spacing and formatting
    match = re.search(r'(Sub\s*Total|Total|Sub-Total|Subtotal)[^\d]*(\d+\.\d{2})', extracted_text, re.IGNORECASE)

    if match:
        total = float(match.group(2))
        return total
    else:
        print("No subtotal or total found in the following text:")
        print(extracted_text)
        return 0.0


def process_receipts(image_paths):
    """Process multiple receipts and extract the total or subtotal for each."""
    totals = []

    for image_path in image_paths:
        image = load_image(image_path)

        # Apply edge detection and transformation
        image = apply_edge_detection_and_transformation(image)

        # Get operations based on the image path
        operations = get_operations_for_image(image_path)

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

        extracted_text = extract_text_from_image(image, lang='eng')

        # Extract the total or subtotal from the text
        total = extract_total_or_subtotal(extracted_text)
        totals.append((image_path, total))

    return totals


def visualize_totals(totals):
    """Visualize the total sales using a bar chart."""
    image_names = [path.split('/')[-1] for path, total in totals]
    total_values = [total for path, total in totals]

    plt.figure(figsize=(10, 6))
    plt.bar(image_names, total_values, color='skyblue')
    plt.xlabel('Receipt Image')
    plt.ylabel('Total Sales ($)')
    plt.title('Total Sales from Receipts')
    plt.xticks(rotation=45, ha="right")
    plt.show()


if __name__ == "__main__":
    image_paths = [
        "images/Recept-I.png",
        "images/Recept-II.png",
        "images/Recept-III.png",
        "images/Recept-IV.png",
        "images/Recepts.png",
    ]

    # Process all receipts and get the totals
    totals = process_receipts(image_paths)

    # Visualize the totals
    visualize_totals(totals)