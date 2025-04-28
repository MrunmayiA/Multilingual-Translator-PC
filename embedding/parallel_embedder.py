import torch
import logging
import numpy as np
from typing import List, Optional
import multiprocessing as mp
from sentence_transformers import SentenceTransformer
from concurrent.futures import ThreadPoolExecutor
import faiss

logger = logging.getLogger(__name__)

class ParallelEmbedder:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2', batch_size: int = 64):
        self.batch_size = batch_size
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"Using device: {self.device}")
        
        # Load model with optimizations
        self.model = SentenceTransformer(model_name)
        if self.device == 'cuda':
            self.model = self.model.half()  # Use FP16 for faster GPU processing
        self.model.to(self.device)
        
        # Initialize FAISS index with optimizations
        self.dimension = self.model.get_sentence_embedding_dimension()
        if self.device == 'cuda':
            # Use IVFFlat index for faster search on GPU
            nlist = 100  # number of clusters
            self.quantizer = faiss.IndexFlatL2(self.dimension)
            self.index = faiss.IndexIVFFlat(self.quantizer, self.dimension, nlist)
            try:
                res = faiss.StandardGpuResources()
                self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
                logger.info("Using GPU-optimized FAISS index")
            except Exception as e:
                logger.warning(f"Using CPU FAISS index: {str(e)}")
                self.index = faiss.IndexFlatL2(self.dimension)
        else:
            self.index = faiss.IndexFlatL2(self.dimension)
    
    @torch.no_grad()
    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """Embed a batch of texts with optimizations."""
        try:
            embeddings = self.model.encode(
                texts,
                batch_size=self.batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,
                device=self.device
            )
            return embeddings
        except Exception as e:
            logger.error(f"Error in batch embedding: {str(e)}")
            return np.array([])
    
    def process_chunks(self, chunks: List[str], num_threads: Optional[int] = None) -> np.ndarray:
        """Process chunks with optimized batching."""
        if not chunks:
            return np.array([])
        
        # Optimize number of threads based on CPU cores and batch size
        if num_threads is None:
            num_threads = min(mp.cpu_count(), max(1, len(chunks) // self.batch_size))
        
        # Pre-allocate memory for embeddings
        total_chunks = len(chunks)
        embeddings_list = []
        
        # Process in larger batches for better GPU utilization
        batch_size = min(self.batch_size * 2, total_chunks)
        batches = [chunks[i:i + batch_size] for i in range(0, total_chunks, batch_size)]
        
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(self.embed_batch, batch) for batch in batches]
            
            for future in futures:
                try:
                    embeddings = future.result()
                    if embeddings.size > 0:
                        embeddings_list.append(embeddings)
                except Exception as e:
                    logger.error(f"Error processing batch: {str(e)}")
        
        if not embeddings_list:
            return np.array([])
        
        return np.vstack(embeddings_list)
    
    def add_to_index(self, embeddings: np.ndarray) -> bool:
        """Add embeddings to FAISS index with training if needed."""
        try:
            if embeddings.size > 0:
                if isinstance(self.index, faiss.IndexIVFFlat) and not self.index.is_trained:
                    logger.info("Training FAISS index...")
                    self.index.train(embeddings)
                self.index.add(embeddings)
                logger.info(f"Added {len(embeddings)} embeddings to index")
                return True
            return False
        except Exception as e:
            logger.error(f"Error adding to index: {str(e)}")
            return False
    
    def embed_and_store(self, chunks: List[str]) -> bool:
        """Optimized embed and store pipeline."""
        try:
            embeddings = self.process_chunks(chunks)
            return self.add_to_index(embeddings)
        except Exception as e:
            logger.error(f"Error in embed_and_store: {str(e)}")
            return False
    
    def search(self, query: str, k: int = 5) -> List[int]:
        """Optimized similarity search."""
        try:
            query_embedding = self.embed_batch([query])
            if query_embedding.size == 0:
                return []
            
            if isinstance(self.index, faiss.IndexIVFFlat):
                # Use more probes for better accuracy with IVFFlat
                self.index.nprobe = min(20, self.index.nlist)
            
            distances, indices = self.index.search(query_embedding, k)
            return indices[0].tolist()
        except Exception as e:
            logger.error(f"Error in search: {str(e)}")
            return [] 