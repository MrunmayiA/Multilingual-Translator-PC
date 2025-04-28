import fitz
import pytesseract
from pdf2image import convert_from_path
import cv2
import numpy as np
from langdetect import detect
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor
import os
import logging
from tqdm import tqdm
import shutil
from typing import List, Tuple, Optional
import queue
import threading

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Cache for opened PDF documents
pdf_cache = {}
pdf_cache_lock = threading.Lock()

def get_pdf_doc(pdf_path: str) -> fitz.Document:
    """Get or create PDF document from cache."""
    with pdf_cache_lock:
        if pdf_path not in pdf_cache:
            pdf_cache[pdf_path] = fitz.open(pdf_path)
        return pdf_cache[pdf_path]

def is_tesseract_available():
    return shutil.which('tesseract') is not None

def process_page_range(pdf_path: str, start_page: int, end_page: int) -> Tuple[int, str]:
    """Process a range of pages efficiently."""
    try:
        doc = get_pdf_doc(pdf_path)
        text = []
        
        # Process pages in the range
        for page_num in range(start_page, end_page):
            page = doc[page_num]
            page_text = page.get_text()
            
            if not page_text.strip() and is_tesseract_available():
                pix = page.get_pixmap()
                img_data = pix.tobytes("png")
                page_text = pytesseract.image_to_string(img_data)
            
            text.append(page_text)
        
        return start_page, "\n".join(text)
    except Exception as e:
        logger.error(f"Error processing pages {start_page}-{end_page}: {str(e)}")
        return start_page, ""

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text from PDF using optimized parallel processing."""
    try:
        doc = get_pdf_doc(pdf_path)
        num_pages = len(doc)
        
        logger.info(f"Starting text extraction from {pdf_path}")
        
        # Optimize chunk size based on available CPU cores
        num_cpus = mp.cpu_count()
        chunk_size = max(5, num_pages // (num_cpus * 2))  # Ensure reasonable chunk size
        chunks = [(i, min(i + chunk_size, num_pages)) 
                 for i in range(0, num_pages, chunk_size)]
        
        # Process chunks in parallel using ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=num_cpus) as executor:
            futures = [
                executor.submit(process_page_range, pdf_path, start, end)
                for start, end in chunks
            ]
            
            # Collect results as they complete
            results = []
            for future in futures:
                try:
                    start_page, text = future.result()
                    if text:
                        results.append((start_page, text))
                except Exception as e:
                    logger.error(f"Error processing chunk: {str(e)}")
        
        # Sort and combine results
        results.sort(key=lambda x: x[0])
        text = "\n".join(text for _, text in results)
        
        logger.info(f"Successfully extracted text from {len(chunks)} chunks")
        return text
        
    except Exception as e:
        logger.error(f"Error in extract_text_from_pdf: {str(e)}")
        return ""
    finally:
        # Clean up PDF cache
        with pdf_cache_lock:
            if pdf_path in pdf_cache:
                pdf_cache[pdf_path].close()
                del pdf_cache[pdf_path]

def extract_text_parallel(pdf_paths: List[str]) -> List[str]:
    """Extract text from multiple PDFs in parallel with optimizations."""
    extracted_texts = []
    
    # Process PDFs in parallel with optimized thread count
    with ThreadPoolExecutor(max_workers=min(len(pdf_paths), mp.cpu_count())) as executor:
        # Submit all PDF processing tasks
        future_to_path = {
            executor.submit(extract_text_from_pdf, path): path 
            for path in pdf_paths
        }
        
        # Process results as they complete
        for future in future_to_path:
            pdf_path = future_to_path[future]
            try:
                text = future.result()
                if text.strip():
                    extracted_texts.append(text)
                    logger.info(f"✅ Extracted text from {pdf_path}")
                    try:
                        lang = detect(text[:1000])
                        logger.info(f"🌐 Detected Language: {lang}")
                    except:
                        logger.warning("⚠️ Could not detect language")
                else:
                    logger.warning(f"⚠️ No text extracted from {pdf_path}")
            except Exception as e:
                logger.error(f"Error processing {pdf_path}: {str(e)}")
    
    return extracted_texts

def extract_text_auto(pdf_path: str) -> str:
    """Extract text from a single PDF using optimized processing."""
    try:
        text = extract_text_from_pdf(pdf_path)
        if text.strip():
            logger.info("✅ Extracted using digital method.")
            try:
                lang = detect(text[:1000])
                logger.info(f"🌐 Detected Language: {lang}")
            except:
                logger.warning("⚠️ Could not detect language")
            return text
        else:
            logger.warning(f"⚠️ No text extracted from {pdf_path}")
            return ""
    except Exception as e:
        logger.error(f"Error in extract_text_auto: {str(e)}")
        return "" 