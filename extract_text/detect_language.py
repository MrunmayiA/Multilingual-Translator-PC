# extract_text/detect_language.py
from langdetect import detect

def detect_language(text):
    try:
        return detect(text)
    except:
        return "unknown"
