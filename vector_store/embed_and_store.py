# ✅ vector_store/embed_and_store.py
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os
import pickle
import torch
from concurrent.futures import ThreadPoolExecutor

def embed_and_store_chunks(chunks, model_name='paraphrase-multilingual-MiniLM-L12-v2', index_path="vector_store/index.faiss", meta_path="vector_store/meta.pkl", batch_size=32):
    print("🔄 Loading embedding model...")
    model = SentenceTransformer(model_name)
    
    # Move model to GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    
    print(f"🧠 Generating embeddings on {device.upper()}...")
    
    # Process in optimized batches for better GPU utilization
    if len(chunks) > batch_size:
        batches = [chunks[i:i+batch_size] for i in range(0, len(chunks), batch_size)]
        print(f"💪 Processing {len(batches)} batches in parallel...")
        
        # Use ThreadPoolExecutor for parallel batch processing
        with ThreadPoolExecutor(max_workers=min(8, os.cpu_count())) as executor:
            batch_embeddings = list(executor.map(lambda batch: model.encode(
                batch, 
                show_progress_bar=False,
                convert_to_numpy=True,
                device=device
            ), batches))
            
        embeddings = np.vstack(batch_embeddings)
    else:
        # Small dataset, process directly
        embeddings = model.encode(chunks, show_progress_bar=True, device=device)

    print("📦 Creating FAISS index...")
    dim = embeddings.shape[1]
    
    # Use GPU for index creation if available
    if torch.cuda.is_available():
        print("🚀 Using GPU acceleration for FAISS...")
        # Create a CPU index first
        cpu_index = faiss.IndexFlatL2(dim)
        
        # Move to GPU for faster processing
        res = faiss.StandardGpuResources()
        gpu_index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
        
        # Add vectors to GPU index
        gpu_index.add(np.array(embeddings).astype(np.float32))
        
        # Move back to CPU for storage
        index = faiss.index_gpu_to_cpu(gpu_index)
    else:
        # CPU fallback
        index = faiss.IndexFlatL2(dim)
        index.add(np.array(embeddings).astype(np.float32))

    os.makedirs(os.path.dirname(index_path), exist_ok=True)
    faiss.write_index(index, index_path)

    with open(meta_path, "wb") as f:
        pickle.dump(chunks, f)

    print(f"✅ Stored {len(chunks)} chunks in vector store!")
