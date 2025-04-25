# chunk_text/chunker.py
import re

def chunk_text(text, chunk_size=3, overlap=1):
    # 🧠 For Bengali or low-punctuation languages, fallback to line-based split
    # Try splitting by sentence punctuation first
    sentences = re.split(r'(?<=[.?!৷।])\s+', text.strip())  # includes Bengali Danda ।

    # If sentence split results in very few sentences, fallback to line split
    if len(sentences) < 5:
        print("⚠️ Very few sentence splits. Falling back to line-based chunking.")
        sentences = [line.strip() for line in text.splitlines() if line.strip()]

    chunks = []
    i = 0
    while i < len(sentences):
        chunk = sentences[i:i + chunk_size]
        chunks.append(" ".join(chunk))
        i += chunk_size - overlap

    print(f"✅ Chunked into {len(chunks)} chunks.")
    return chunks
