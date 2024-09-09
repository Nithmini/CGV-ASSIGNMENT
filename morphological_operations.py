import cv2
import numpy as np

# Load the Binarized Image
image_path = 'images/Recept-II.png'
binary_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Create a kernel for morphological operations
kernel = np.ones((3, 3), np.uint8)

# Perform Dilation (to enhance text)
dilated_image = cv2.dilate(binary_image, kernel, iterations=1)

# Perform Erosion (to remove noise)
eroded_image = cv2.erode(dilated_image, kernel, iterations=1)

# Show the Morphological Operations Result
cv2.imshow('Dilated Image', dilated_image)
cv2.imshow('Eroded Image', eroded_image)
cv2.waitKey(0)
cv2.destroyAllWindows()