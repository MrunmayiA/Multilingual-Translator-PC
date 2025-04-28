# Optimized Text Processing and Embedding System

This system provides a highly optimized solution for processing text data and generating embeddings using state-of-the-art transformer models.

## Features

- **GPU Acceleration**: Automatic GPU utilization for both model inference and FAISS indexing
- **Batched Processing**: Dynamic batch sizing based on available resources
- **Parallel Processing**: Efficient multi-threading for concurrent batch processing
- **FAISS Integration**: Fast similarity search with GPU acceleration support
- **Memory Optimization**: Controlled resource usage and efficient array operations
- **Robust Error Handling**: Comprehensive error handling and detailed logging

## Requirements

- Python 3.8+
- CUDA-compatible GPU (optional, for GPU acceleration)
- Dependencies listed in `requirements.txt`

## Installation

```bash
pip install -r requirements.txt
```

For GPU support with FAISS:
```bash
pip uninstall faiss-cpu
pip install faiss-gpu
```

## Usage

```python
from embedding.parallel_embedder import ParallelEmbedder

# Initialize the embedder
embedder = ParallelEmbedder()

# Process text chunks
chunks = ["your", "text", "chunks", "here"]
embedder.embed_and_store_chunks(chunks)

# Search for similar chunks
query = "your search query"
similar_chunks = embedder.search_similar_chunks(query, k=5)
```

## Performance Optimization Tips

1. **GPU Usage**: 
   - Ensure CUDA is properly installed for GPU acceleration
   - Monitor GPU memory usage for optimal batch sizes

2. **Batch Size**:
   - Default batch size is optimized for most use cases
   - Adjust based on your specific memory constraints

3. **Thread Count**:
   - Automatically optimized based on CPU count
   - Can be manually adjusted if needed

## Error Handling

The system includes comprehensive error handling and logging:
- All operations are logged to help with debugging
- Errors are caught and handled gracefully
- Failed operations are reported with detailed error messages

## Contributing

Feel free to submit issues and enhancement requests! 