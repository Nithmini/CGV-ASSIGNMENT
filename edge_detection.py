import cv2


def apply_edge_detection(image_path, low_threshold=50, high_threshold=150):
    """Apply Canny Edge Detection to an image loaded from the specified path."""
    image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray_image, low_threshold, high_threshold)
    return edges


if __name__ == "__main__":
    image_path = 'images/Recept-II.png'
    edges = apply_edge_detection(image_path)
    cv2.imshow('Edges', edges)
    cv2.waitKey(0)
    cv2.destroyAllWindows()