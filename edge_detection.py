import cv2
import numpy as np

def apply_clahe(image, clip_limit=3.0, tile_grid_size=(8, 8)):
    """Apply CLAHE to enhance the contrast of the image."""
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    cl1 = clahe.apply(gray_image)
    return cl1

def apply_adaptive_threshold(image, block_size=11, C=8):
    """Apply adaptive thresholding."""
    adaptive_thresh = cv2.adaptiveThreshold(
        image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, block_size, C)
    return adaptive_thresh

def apply_morphology(binary_image, kernel_size=(5, 5), iterations=2):
    """Apply morphological operations to enhance the binary image."""
    kernel = np.ones(kernel_size, np.uint8)
    closing = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel, iterations=iterations)
    dilated = cv2.dilate(closing, kernel, iterations=1)
    return dilated

def find_receipt_contours(binary_image, min_area_ratio=0.01, max_area_ratio=0.95, min_height=200):
    """Find all contours that could represent receipts in the image."""
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h, w = binary_image.shape[:2]
    min_area = min_area_ratio * h * w
    max_area = max_area_ratio * h * w

    valid_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        x, y, w, h = cv2.boundingRect(contour)

        # Filter based on area size and minimum height
        if min_area < area < max_area and h >= min_height:
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            valid_contours.append((x, y, w, h))
    return valid_contours

def combine_overlapping_rectangles(contours):
    """Combine overlapping rectangles."""
    if not contours:
        return []

    # Sort contours by the x coordinate
    contours = sorted(contours, key=lambda x: x[0])

    combined_rectangles = []
    current = contours[0]

    for next_rect in contours[1:]:
        x1, y1, w1, h1 = current
        x2, y2, w2, h2 = next_rect

        # Check if rectangles overlap
        if x1 < x2 + w2 and x1 + w1 > x2 and y1 < y2 + h2 and y1 + h1 > y2:
            # Combine the rectangles
            new_x = min(x1, x2)
            new_y = min(y1, y2)
            new_w = max(x1 + w1, x2 + w2) - new_x
            new_h = max(y1 + h1, y2 + h2) - new_y
            current = (new_x, new_y, new_w, new_h)
        else:
            combined_rectangles.append(current)
            current = next_rect

    # Add the last rectangle
    combined_rectangles.append(current)

    return combined_rectangles

def draw_bounding_boxes(image, contours, color=(0, 255, 0), thickness=3):
    """Draw bounding boxes around all detected contours."""
    image_with_boxes = image.copy()
    for (x, y, w, h) in contours:
        cv2.rectangle(image_with_boxes, (x, y), (x + w, y + h), color, thickness)
    return image_with_boxes

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
    contours = find_receipt_contours(morphed_image, min_height=200)

    # Step 4: Combine overlapping rectangles
    combined_contours = combine_overlapping_rectangles(contours)

    # Step 5: Draw bounding boxes around each detected receipt
    image_with_boxes = draw_bounding_boxes(image, combined_contours)

    # Display the results
    cv2.imshow('Original Image', image)
    cv2.imshow('Binary Image', binary_image)
    cv2.imshow('Image with Bounding Boxes', image_with_boxes)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()