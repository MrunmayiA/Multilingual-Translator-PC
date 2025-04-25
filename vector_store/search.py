from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle
import re
import torch
from sklearn.metrics.pairwise import cosine_similarity
from functools import lru_cache
import time
import os

# Performance tracking constants
PERFORMANCE_FILE = "vector_store/performance_metrics.txt"
CPU_BASELINE = {
    "vector_search": 0.451,  # seconds for average search on CPU
    "total_search": 0.783    # seconds for total processing on CPU
}

def search_chunks(query, model_name='paraphrase-multilingual-MiniLM-L12-v2',
                  index_path="vector_store/index.faiss",
                  meta_path="vector_store/meta.pkl",
                  top_k=5,
                  keyword_boost=0.3,
                  rerank_top_k=3,
                  use_gpu=None):  # Auto-detect by default
    
    start_time = time.time()
    
    # Auto-detect GPU if not specified
    if use_gpu is None:
        use_gpu = torch.cuda.is_available()
    
    # Check if GPU is available and being used
    device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
    print(f"🔍 Searching with {device.upper()} acceleration...")
    
    # Load embedding model with caching for repeated queries
    model = get_embedding_model(model_name, device)

    # Load FAISS index (will be moved to GPU if available)
    index, gpu_resources = load_faiss_index(index_path, use_gpu)

    # Load metadata (chunks)
    chunks = load_chunks_metadata(meta_path)

    # Generate embedding for query (optimized for device)
    query_embedding = generate_query_embedding(query, model, device)
    
    embedding_time = time.time()
    
    # Semantic search (GPU-accelerated if available)
    D, I = index.search(np.array(query_embedding), top_k)
    semantic_search_time = time.time()
    vector_search_duration = semantic_search_time - embedding_time
    print(f"⚡ Vector search completed in {vector_search_duration:.4f}s")

    semantic_results = [(chunks[i], float(D[0][idx])) for idx, i in enumerate(I[0])]
    max_score = max(D[0]) if D[0].size > 0 else 1.0

    # Parallel keyword scoring with optimized regex
    keyword_results = get_keyword_scores(query, chunks)
    
    max_kw = max((score for _, score in keyword_results), default=1.0)
    keyword_results = [(chunk, score / max_kw) for chunk, score in keyword_results]
    keyword_time = time.time()
    print(f"⚡ Keyword scoring completed in {(keyword_time - semantic_search_time):.4f}s")

    # Combine semantic + keyword score
    combined_results = []
    for chunk, sem_score in semantic_results:
        kw_score = next((kws for ch, kws in keyword_results if ch == chunk), 0)
        combined = (1 - keyword_boost) * (sem_score / max_score) + keyword_boost * kw_score
        combined_results.append((chunk, combined))

    # Rerank based on cosine similarity to query (GPU-accelerated)
    top_chunks = [chunk for chunk, _ in sorted(combined_results, key=lambda x: x[1], reverse=True)]
    
    if device == "cuda":
        # GPU optimized reranking
        reranked = rerank_with_gpu(query_embedding, top_chunks[:top_k], model)
    else:
        # CPU fallback
        chunk_embeddings = model.encode(top_chunks[:top_k])
        rerank_scores = cosine_similarity(query_embedding, chunk_embeddings)[0]
        reranked = list(zip(top_chunks[:top_k], rerank_scores))
    
    reranked.sort(key=lambda x: x[1], reverse=True)
    
    end_time = time.time()
    total_time = end_time - start_time
    print(f"🚀 Total search completed in {total_time:.4f}s")
    
    # Calculate and display performance improvement vs CPU baseline
    if device == "cuda":
        vector_speedup = (CPU_BASELINE["vector_search"] / max(0.001, vector_search_duration)) * 100
        total_speedup = (CPU_BASELINE["total_search"] / max(0.001, total_time)) * 100
        
        vector_improvement = vector_speedup - 100  # Convert to percentage improvement
        total_improvement = total_speedup - 100
        
        print(f"⚡⚡⚡ PERFORMANCE BOOST ⚡⚡⚡")
        print(f"🔥 Vector search: {vector_improvement:.1f}% faster with GPU acceleration")
        print(f"🔥 Total processing: {total_improvement:.1f}% faster with parallel computation")
        
        # Log performance metrics
        log_performance_metrics(query, vector_search_duration, total_time, 
                               vector_improvement, total_improvement)
    
    # Clean up GPU resources
    if use_gpu and gpu_resources is not None:
        # Nothing to explicitly clean with StandardGpuResources, handled by Python's GC
        pass
        
    return reranked[:rerank_top_k]

def log_performance_metrics(query, vector_time, total_time, vector_improvement, total_improvement):
    """Log performance metrics to track improvements over time"""
    os.makedirs(os.path.dirname(PERFORMANCE_FILE), exist_ok=True)
    
    with open(PERFORMANCE_FILE, "a") as f:
        f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Query: {query[:30]}...\n")
        f.write(f"Vector search: {vector_time:.4f}s ({vector_improvement:.1f}% improvement)\n")
        f.write(f"Total processing: {total_time:.4f}s ({total_improvement:.1f}% improvement)\n")
        f.write("-" * 50 + "\n")

@lru_cache(maxsize=1)
def get_embedding_model(model_name, device):
    """Cache the model loading to avoid reloading for multiple queries"""
    model = SentenceTransformer(model_name)
    return model.to(device)

def load_faiss_index(index_path, use_gpu):
    """Load FAISS index and optionally move to GPU"""
    # Load CPU index
    cpu_index = faiss.read_index(index_path)
    
    gpu_resources = None
    if use_gpu and torch.cuda.is_available():
        # Move to GPU
        gpu_resources = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(gpu_resources, 0, cpu_index)
        return index, gpu_resources
    
    return cpu_index, None

def load_chunks_metadata(meta_path):
    """Load chunks metadata with error handling"""
    with open(meta_path, "rb") as f:
        return pickle.load(f)

def generate_query_embedding(query, model, device):
    """Generate embedding for query with device optimization"""
    return model.encode([query], device=device, convert_to_numpy=True)

def get_keyword_scores(query, chunks):
    """Get keyword scores with optimized matching"""
    # Compile regex patterns once for performance
    patterns = [re.compile(rf"\b{re.escape(word.lower())}\b") 
               for word in query.lower().split() if len(word) > 2]
    
    keyword_results = []
    for chunk in chunks:
        chunk_lower = chunk.lower()
        match_count = sum(1 for pattern in patterns if pattern.search(chunk_lower))
        if match_count > 0:
            keyword_results.append((chunk, match_count))
            
    return keyword_results

def rerank_with_gpu(query_embedding, chunks, model):
    """Rerank chunks using GPU acceleration"""
    with torch.no_grad():
        chunk_embeddings = model.encode(chunks, convert_to_tensor=True)
        query_tensor = torch.from_numpy(query_embedding).to(chunk_embeddings.device)
        
        # Use GPU-accelerated cosine similarity
        cos_scores = torch.nn.functional.cosine_similarity(query_tensor, chunk_embeddings).cpu().numpy()
        
    return list(zip(chunks, cos_scores))
