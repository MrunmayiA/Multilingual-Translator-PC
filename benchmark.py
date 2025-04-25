#!/usr/bin/env python
"""
GPU-Accelerated RAG Benchmark
-----------------------------
This script compares the performance of CPU vs GPU processing
for the PDF RAG system's most intensive operations.
"""

import time
import numpy as np
import matplotlib.pyplot as plt
from extract_text.extractor import extract_text_auto
from chunk_text.chunker import chunk_text
from vector_store.embed_and_store import embed_and_store_chunks
from vector_store.search import search_chunks
import torch
import os

BENCHMARK_RESULTS = "benchmark_results.txt"
SAMPLE_PDF = "pdf_samples/sample2.pdf"  # Adjust this to your sample PDF
SAMPLE_QUERIES = [
    "What is the main topic of this document?",
    "Explain the key concepts mentioned in the introduction",
    "Summarize the conclusion and findings",
    "What methods were used in the research?",
    "Who are the main authors cited in this document?"
]

def log_result(operation, cpu_time, gpu_time):
    """Log benchmark results to file and console"""
    if gpu_time > 0:
        improvement = ((cpu_time / gpu_time) - 1) * 100
    else:
        improvement = 0
        
    result = f"{operation}: CPU={cpu_time:.4f}s, GPU={gpu_time:.4f}s, Improvement={improvement:.1f}%"
    
    with open(BENCHMARK_RESULTS, "a") as f:
        f.write(result + "\n")
    
    print(result)
    return improvement

def benchmark_embedding(chunks):
    """Benchmark embedding generation performance"""
    print("\n🔍 Benchmarking embedding generation...")
    
    # Force CPU embedding
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    t0 = time.time()
    embed_and_store_chunks(chunks, index_path="vector_store/cpu_index.faiss")
    cpu_time = time.time() - t0
    
    # Enable GPU embedding if available
    os.environ["CUDA_VISIBLE_DEVICES"] = "0" if torch.cuda.is_available() else ""
    t0 = time.time()
    embed_and_store_chunks(chunks, index_path="vector_store/gpu_index.faiss")
    gpu_time = time.time() - t0
    
    return log_result("Embedding generation", cpu_time, gpu_time)

def benchmark_search():
    """Benchmark search performance"""
    print("\n🔍 Benchmarking vector search...")
    
    # Prepare for search benchmarking
    cpu_times = []
    gpu_times = []
    
    for query in SAMPLE_QUERIES:
        print(f"\nQuery: {query}")
        
        # Force CPU search
        t0 = time.time()
        search_chunks(query, use_gpu=False)
        cpu_time = time.time() - t0
        cpu_times.append(cpu_time)
        
        # Enable GPU search if available
        if torch.cuda.is_available():
            t0 = time.time()
            search_chunks(query, use_gpu=True)
            gpu_time = time.time() - t0
        else:
            gpu_time = 0
        gpu_times.append(gpu_time)
    
    avg_cpu = np.mean(cpu_times)
    avg_gpu = np.mean(gpu_times) if any(gpu_times) else 0
    
    return log_result("Vector search (average)", avg_cpu, avg_gpu)

def create_performance_chart(embedding_improvement, search_improvement):
    """Create a bar chart showing performance improvements"""
    operations = ['Embedding Generation', 'Vector Search']
    improvements = [embedding_improvement, search_improvement]
    
    plt.figure(figsize=(10, 6))
    plt.bar(operations, improvements, color=['#2C82C9', '#EF4836'])
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.title('GPU Acceleration Performance Improvement', fontsize=15)
    plt.ylabel('Improvement (%)', fontsize=12)
    plt.grid(axis='y', alpha=0.3)
    
    # Add labels on top of bars
    for i, v in enumerate(improvements):
        plt.text(i, v + 5, f"{v:.1f}%", ha='center', fontweight='bold')
    
    plt.savefig('performance_improvement.png')
    print("\n✅ Performance chart saved to 'performance_improvement.png'")

def main():
    """Run the full benchmark suite"""
    print("🚀 Starting RAG Performance Benchmark")
    print(f"GPU Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU Device: {torch.cuda.get_device_name(0)}")
    
    # Clear previous results
    if os.path.exists(BENCHMARK_RESULTS):
        os.remove(BENCHMARK_RESULTS)
    
    # Extract text from sample PDF
    print("\n📄 Extracting text from sample PDF...")
    text = extract_text_auto(SAMPLE_PDF)
    
    # Chunk the text
    print("\n✂️ Chunking text...")
    chunks = chunk_text(text)
    print(f"Created {len(chunks)} chunks")
    
    # Run benchmarks
    embedding_improvement = benchmark_embedding(chunks)
    search_improvement = benchmark_search()
    
    # Create visualization
    if embedding_improvement > 0 and search_improvement > 0:
        create_performance_chart(embedding_improvement, search_improvement)
    
    print("\n✅ Benchmark complete!")
    print(f"Detailed results saved to {BENCHMARK_RESULTS}")

if __name__ == "__main__":
    main() 