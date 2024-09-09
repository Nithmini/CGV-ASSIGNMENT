import cv2
import pytesseract

def extract_text_from_image(image, lang='eng'):
    """Extract text from a given image using Tesseract OCR."""
    # Convert the image to grayscale if it's not already
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Use Tesseract to extract text
    custom_config = r'--oem 3 --psm 6'  # Configuration options for Tesseract
    text = pytesseract.image_to_string(image, config=custom_config, lang=lang)

    return text

def extract_text_from_receipts(transformed_receipts, lang='eng'):
    """Extract text from a list of transformed receipts."""
    for i, receipt in enumerate(transformed_receipts):
        print(f"Extracting text from receipt {i + 1}...")
        text = extract_text_from_image(receipt, lang=lang)
        print(f"Text from receipt {i + 1}:\n{text}\n{'-' * 50}\n")


if __name__ == "__main__":
    transformed_receipts = [cv2.imread("images/Recept-I.png")] 

    # Extract text from each receipt and print it to the terminal
    extract_text_from_receipts(transformed_receipts, lang='eng')