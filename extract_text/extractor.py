# extract_text/extractor.py
from .extract_digital import extract_text_from_digital_pdf
from .extract_scanned import extract_text_from_scanned_pdf
from .detect_language import detect_language

def extract_text_auto(pdf_path):
    text = extract_text_from_digital_pdf(pdf_path)
    if len(text.strip()) < 100:
        print("❗️Digital extraction failed. Using OCR...")
        text = extract_text_from_scanned_pdf(pdf_path)
    else:
        print("✅ Extracted using digital method.")

    lang = detect_language(text)
    print(f"🌐 Detected Language: {lang}")
    return text
