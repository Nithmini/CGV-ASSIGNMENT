import cv2

# Load the image
image_path = 'images/Recept-I.png'
image = cv2.imread(image_path)

# Convert to Grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Show the Grayscale Image
cv2.imshow('Grayscale Image', gray_image)
cv2.waitKey(0)
cv2.destroyAllWindows()