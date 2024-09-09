import cv2
import numpy as np
from edge_detection import (
    apply_clahe,
    apply_adaptive_threshold,
    apply_morphology,
    find_receipt_contours,
    remove_nested_rectangles
)


def get_perspective_transform(image, bounding_box):
    """Apply perspective transformation to get a top-down view of the receipt."""
    x, y, w, h = bounding_box

    # Create a contour from the bounding box
    rect_points = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], dtype="float32")

    # Define the destination points for the top-down view
    dst = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype="float32")

    # Compute the perspective transform matrix and apply it
    M = cv2.getPerspectiveTransform(rect_points, dst)
    warped = cv2.warpPerspective(image, M, (w, h))

    return warped


def draw_transformed_receipts(image, bounding_boxes):
    """Transform and display all detected receipts."""
    transformed_receipts = []

    for i, bounding_box in enumerate(bounding_boxes):
        # Apply perspective transformation
        transformed_receipt = get_perspective_transform(image, bounding_box)
        transformed_receipts.append(transformed_receipt)

        # Display each transformed receipt in a unique window
        window_name = f'Transformed Receipt {i+1}'
        cv2.imshow(window_name, transformed_receipt)

    # Wait for a key press to close all windows
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return transformed_receipts


def main():
    image_path = 'images/Recepts.png'
    image = cv2.imread(image_path)
    if image is None:
        print("Image not found.")
        return

    # Step 1: Enhance contrast and apply adaptive thresholding
    cl1 = apply_clahe(image)
    binary_image = apply_adaptive_threshold(cl1)

    # Step 2: Apply morphological operations
    morphed_image = apply_morphology(binary_image)

    # Step 3: Detect all contours that could be receipts, with a minimum height of 200 pixels
    bounding_boxes = find_receipt_contours(morphed_image, min_height=200)

    # Step 4: Remove nested rectangles (now removes nested bounding boxes)
    non_nested_boxes = remove_nested_rectangles(bounding_boxes)

    # Step 5: Apply the perspective transformation to each detected receipt and display them
    draw_transformed_receipts(image, non_nested_boxes)


if __name__ == "__main__":
    main()