from extract_text.extractor import extract_text_auto
from chunk_text.chunker import chunk_text
from vector_store.embed_and_store import embed_and_store_chunks
from vector_store.search import search_chunks
from llm.answer_generator import generate_answer, decompose_query  # Local version using Phi-2


# Step 1: Extract text from the PDF
pdf_file = "pdf_samples/sample2.pdf"
text = extract_text_auto(pdf_file)

# Step 2: Chunk the extracted text
chunks = chunk_text(text, chunk_size=500, overlap=100)
print(f"\n🧩 Total chunks: {len(chunks)}")

# Step 3: Embed and store the chunks in FAISS
embed_and_store_chunks(chunks)

# Step 4: Ask the user for a question
query = input("\n🔎 Ask a question based on the PDF: ")

# Step 5: Search for matching chunks using vector similarity
results = search_chunks(query)

# Step 6: Show the top matches (trimmed)
print("\n🎯 Top Matching Chunks:\n")
for i, (chunk, score) in enumerate(results, 1):
    snippet = chunk[:300].strip().replace("\n", " ") + "..."
    print(f"--- Result {i} (Score: {score:.2f}) ---\n{snippet}\n")

# Step 7: Generate the final answer using a local model (phi-2)
top_chunks = [chunk for chunk, score in results[:2]]
final_answer = generate_answer(query, top_chunks)

# Step 8: Show the final answer
print("\n🧠 Final Answer:\n")
print(final_answer)

def rag_query(question, chat_history):
    sub_questions = decompose_query(question)

    combined_answer = ""
    for sq in sub_questions:
        results = search_chunks(sq)
        top_chunks = [chunk for chunk, score in results[:2]]
        answer = generate_answer(sq, top_chunks)
        combined_answer += f"\nQ: {sq}\nA: {answer}\n"

    # Update chat history
    chat_history.append((question, combined_answer.strip()))
    return chat_history, chat_history