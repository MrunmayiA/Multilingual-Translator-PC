import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from typing import List, Tuple
import logging
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def split_into_sentences(text: str) -> List[str]:
    """Split text into sentences using regex patterns"""
    # Split on period followed by space and uppercase letter
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
    return [s.strip() for s in sentences if s.strip()]

def process_text_segment(args: Tuple[str, int, int, int]) -> List[str]:
    """Process a segment of text into chunks"""
    text_segment, chunk_size, overlap, segment_id = args
    try:
        sentences = split_into_sentences(text_segment)
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence_length = len(sentence)
            
            if current_length + sentence_length > chunk_size and current_chunk:
                # Store the current chunk
                chunks.append(" ".join(current_chunk))
                # Keep last sentences for overlap
                overlap_size = 0
                overlap_chunk = []
                for s in reversed(current_chunk):
                    if overlap_size + len(s) <= overlap:
                        overlap_chunk.insert(0, s)
                        overlap_size += len(s)
                    else:
                        break
                current_chunk = overlap_chunk
                current_length = sum(len(s) for s in current_chunk)
            
            current_chunk.append(sentence)
            current_length += sentence_length
        
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        return chunks
    except Exception as e:
        logger.error(f"Error processing segment {segment_id}: {str(e)}")
        return []

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 100) -> List[str]:
    """Chunk text into smaller segments using parallel processing"""
    try:
        # Calculate optimal segment size based on CPU count
        num_workers = min(32, mp.cpu_count() * 2)
        approx_segments = max(num_workers, len(text) // (chunk_size * 2))
        segment_size = len(text) // approx_segments
        
        # Create segments with overlap to handle sentence boundaries
        segments = []
        for i in range(approx_segments):
            start = max(0, i * segment_size - overlap)
            end = len(text) if i == approx_segments - 1 else (i + 1) * segment_size + overlap
            segments.append((text[start:end], chunk_size, overlap, i))
        
        # Process segments in parallel using ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(process_text_segment, segment) for segment in segments]
            chunk_lists = [f.result() for f in futures]
        
        # Combine chunks from all segments and remove duplicates
        all_chunks = []
        seen = set()
        for chunk_list in chunk_lists:
            for chunk in chunk_list:
                if chunk and chunk not in seen:
                    seen.add(chunk)
                    all_chunks.append(chunk)
        
        logger.info(f"Successfully chunked text into {len(all_chunks)} chunks")
        return all_chunks
    except Exception as e:
        logger.error(f"Error in chunk_text: {str(e)}")
        return [] 