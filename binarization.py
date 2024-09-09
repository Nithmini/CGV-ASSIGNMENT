import cv2

# Load the Grayscale Image
image_path = 'images/Recept-II.png'
gray_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Binarization (Thresholding)
_, binary_image = cv2.threshold(gray_image, 150, 255, cv2.THRESH_BINARY_INV)

# Show the Binarized Image
cv2.imshow('Binarized Image', binary_image)
cv2.waitKey(0)
cv2.destroyAllWindows()