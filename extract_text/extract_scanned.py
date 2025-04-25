# extract_text/extract_scanned.py

from pdf2image import convert_from_path
import pytesseract
import cv2
import os

# Set the path to your Poppler binaries (adjust this path if needed)
POPLER_PATH = r"C:\Program Files\poppler-24.08.0\Library\bin"

def extract_text_from_scanned_pdf(pdf_path, lang='eng'):
    # Use Poppler path to convert scanned PDF to images
    pages = convert_from_path(pdf_path, poppler_path=POPLER_PATH)
    full_text = ""

    for i, page in enumerate(pages):
        img_path = f"temp_page_{i}.jpg"
        page.save(img_path, "JPEG")

        img = cv2.imread(img_path)
        text = pytesseract.image_to_string(img, lang=lang)
        full_text += text + "\n"

        os.remove(img_path)  # Clean up temp file

    return full_text
