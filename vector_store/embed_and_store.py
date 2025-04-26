from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os
import pickle
from concurrent.futures import ThreadPoolExecutor
import time

def embed_and_store_chunks(chunks, model_name='paraphrase-multilingual-MiniLM-L12-v2', index_path="vector_store/index.faiss", meta_path="vector_store/meta.pkl", batch_size=256):
    print("🔄 Loading embedding model...")
    model = SentenceTransformer(model_name)
    model = model.to('cpu')
    device = 'cpu'

    print(f"🧠 Generating embeddings on CPU...")

    start_time = time.perf_counter()

    def encode_batch(batch):
        return model.encode(
            batch,
            batch_size=64,  # Important: internal batch processing
            show_progress_bar=False,
            convert_to_numpy=True,
            device=device,
            normalize_embeddings=True  # Important: faster for FAISS
        )

    if len(chunks) > batch_size:
        batches = [chunks[i:i+batch_size] for i in range(0, len(chunks), batch_size)]
        print(f"💪 Processing {len(batches)} batches with ThreadPoolExecutor...")

        with ThreadPoolExecutor(max_workers=min(4, os.cpu_count())) as executor:
            batch_embeddings = list(executor.map(encode_batch, batches))

        embeddings = np.vstack(batch_embeddings)
    else:
        embeddings = model.encode(
            chunks,
            batch_size=64,
            show_progress_bar=True,
            convert_to_numpy=True,
            device=device,
            normalize_embeddings=True
        )

    end_time = time.perf_counter()
    print(f"✅ Embedding completed in {end_time - start_time:.2f} seconds.")

    print("📦 Creating FAISS index...")
    dim = embeddings.shape[1]

    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings).astype(np.float32))

    os.makedirs(os.path.dirname(index_path), exist_ok=True)
    faiss.write_index(index, index_path)

    with open(meta_path, "wb") as f:
        pickle.dump(chunks, f)

    print(f"✅ Stored {len(chunks)} chunks in vector store!")
