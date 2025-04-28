import torch
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor
from typing import List
import pickle
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ParallelEmbedder:
    def __init__(self, model_name='all-MiniLM-L6-v2', batch_size=32):
        self.model = SentenceTransformer(model_name)
        self.batch_size = batch_size
        self.dimension = self.model.get_sentence_embedding_dimension()
        
        # Move model to GPU if available
        if torch.cuda.is_available():
            self.model = self.model.to(torch.device('cuda'))
            logger.info("Using GPU for embeddings generation")
        else:
            logger.info("Using CPU for embeddings generation")
    
    def embed_batch(self, batch: List[str]) -> np.ndarray:
        """Embed a batch of text chunks"""
        try:
            with torch.no_grad():
                embeddings = self.model.encode(
                    batch,
                    batch_size=self.batch_size,
                    show_progress_bar=False,
                    convert_to_tensor=True
                )
                return embeddings.cpu().numpy()
        except Exception as e:
            logger.error(f"Error embedding batch: {str(e)}")
            return np.array([])

def create_batches(chunks: List[str], batch_size: int) -> List[List[str]]:
    """Create batches of chunks for parallel processing"""
    return [chunks[i:i + batch_size] for i in range(0, len(chunks), batch_size)]

def embed_and_store_chunks(chunks: List[str], batch_size: int = 32):
    """Embed and store text chunks using parallel processing and GPU acceleration"""
    try:
        # Initialize embedder
        embedder = ParallelEmbedder(batch_size=batch_size)
        
        # Create batches
        batches = create_batches(chunks, batch_size)
        
        # Initialize FAISS index
        index = faiss.IndexFlatL2(embedder.dimension)
        
        # Try to use GPU for FAISS if available
        try:
            if torch.cuda.is_available():
                # Check if we have the GPU version of FAISS
                if hasattr(faiss, 'StandardGpuResources'):
                    res = faiss.StandardGpuResources()
                    index = faiss.index_cpu_to_gpu(res, 0, index)
                    logger.info("Using GPU for FAISS index")
                else:
                    logger.info("FAISS GPU support not available, using CPU")
        except Exception as e:
            logger.warning(f"Could not initialize GPU for FAISS: {str(e)}")
        
        # Process batches with progress bar
        all_embeddings = []
        for batch in tqdm(batches, desc="Generating embeddings"):
            embeddings = embedder.embed_batch(batch)
            if len(embeddings) > 0:
                all_embeddings.append(embeddings)
        
        if all_embeddings:
            # Combine all embeddings
            final_embeddings = np.vstack(all_embeddings)
            
            # Add to index
            index.add(final_embeddings)
            
            # Save index and chunks
            os.makedirs('vector_store', exist_ok=True)
            
            # Convert GPU index to CPU for saving if necessary
            try:
                if hasattr(faiss, 'index_gpu_to_cpu') and torch.cuda.is_available():
                    index = faiss.index_gpu_to_cpu(index)
            except Exception as e:
                logger.warning(f"Error converting GPU index to CPU: {str(e)}")
            
            faiss.write_index(index, 'vector_store/index.faiss')
            with open('vector_store/chunks.pkl', 'wb') as f:
                pickle.dump(chunks, f)
            
            logger.info(f"Successfully embedded and stored {len(chunks)} chunks")
            return index
        else:
            logger.error("No valid embeddings generated")
            return None
            
    except Exception as e:
        logger.error(f"Error in embed_and_store_chunks: {str(e)}")
        return None 