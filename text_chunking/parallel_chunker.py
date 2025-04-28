import re
import logging
from typing import List, Tuple
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from itertools import islice

logger = logging.getLogger(__name__)

def batch_texts(texts: List[str], batch_size: int):
    """Yield batches of texts for more efficient parallel processing."""
    it = iter(texts)
    while batch := list(islice(it, batch_size)):
        yield batch

def chunk_text(text: str, min_chunk_size: int = 100, max_chunk_size: int = 1000) -> List[str]:
    """Chunk a single text into smaller pieces."""
    # Split into sentences first
    sentence_pattern = r'(?<=[.!?])\s+'
    sentences = re.split(sentence_pattern, text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    chunks = []
    current_chunk = []
    current_size = 0
    
    for sentence in sentences:
        sentence_size = len(sentence)
        
        # If single sentence is larger than max_chunk_size, split it
        if sentence_size > max_chunk_size:
            if current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
                current_size = 0
            
            # Split long sentence by punctuation or spaces
            sub_sentences = re.split(r'[,;:\s]\s*', sentence)
            sub_chunk = []
            sub_size = 0
            
            for sub_sent in sub_sentences:
                if sub_size + len(sub_sent) > max_chunk_size:
                    if sub_chunk:
                        chunks.append(" ".join(sub_chunk))
                    sub_chunk = [sub_sent]
                    sub_size = len(sub_sent)
                else:
                    sub_chunk.append(sub_sent)
                    sub_size += len(sub_sent)
            
            if sub_chunk:
                chunks.append(" ".join(sub_chunk))
            continue
        
        # Normal sentence processing
        if current_size + sentence_size > max_chunk_size:
            if current_chunk:
                chunks.append(" ".join(current_chunk))
            current_chunk = [sentence]
            current_size = sentence_size
        else:
            current_chunk.append(sentence)
            current_size += sentence_size
        
        # If we've reached a reasonable chunk size, save it
        if current_size >= min_chunk_size:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
            current_size = 0
    
    # Add any remaining text
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
    return chunks

def process_batch(batch: List[str], min_chunk_size: int = 100, max_chunk_size: int = 1000) -> List[str]:
    """Process a batch of texts in a single process."""
    all_chunks = []
    for text in batch:
        chunks = chunk_text(text, min_chunk_size, max_chunk_size)
        all_chunks.extend(chunks)
    return all_chunks

def chunk_texts_parallel(texts: List[str], min_chunk_size: int = 100, max_chunk_size: int = 1000) -> List[str]:
    """Chunk multiple texts in parallel using batch processing."""
    if not texts:
        return []
    
    num_cpus = mp.cpu_count()
    batch_size = max(1, len(texts) // (num_cpus * 2))  # Ensure at least 2 batches per CPU
    all_chunks = []
    
    with ProcessPoolExecutor(max_workers=num_cpus) as executor:
        futures = []
        for batch in batch_texts(texts, batch_size):
            future = executor.submit(process_batch, batch, min_chunk_size, max_chunk_size)
            futures.append(future)
        
        for future in futures:
            try:
                chunks = future.result()
                all_chunks.extend(chunks)
            except Exception as e:
                logger.error(f"Error in chunk processing: {str(e)}")
    
    logger.info(f"Created {len(all_chunks)} chunks from {len(texts)} texts")
    return all_chunks 