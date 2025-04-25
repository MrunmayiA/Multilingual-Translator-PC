import gradio as gr
from extract_text.extractor import extract_text_auto
from chunk_text.chunker import chunk_text
from vector_store.embed_and_store import embed_and_store_chunks
from vector_store.search import search_chunks
from llm.answer_generator import generate_answer

chat_history = []

# PDF processing
def process_pdf(pdf_file):
    path = pdf_file.name
    text = extract_text_auto(path)
    chunks = chunk_text(text, chunk_size=500, overlap=100)
    embed_and_store_chunks(chunks)
    return f"✅ PDF processed with {len(chunks)} chunks."

# Chat-based RAG
def rag_query(question, reference):
    global chat_history
    results = search_chunks(question)
    top_chunks = [chunk for chunk, _ in results[:2]]

    memory_context = "\n".join([f"Q: {q}\nA: {a}" for q, a in chat_history])
    context_chunks = [memory_context] + top_chunks if memory_context else top_chunks

    answer, rouge_score = generate_answer(question, context_chunks, reference=reference)

    # Append score to answer if reference was provided
    if reference and rouge_score:
        answer += f"\n\n🧪 ROUGE-L Score: {rouge_score['rougeL']:.2f}"

    chat_history.append((question, answer))

    # Format for chatbot display
    messages = []
    for q, a in chat_history:
        messages.append({"role": "user", "content": q})
        messages.append({"role": "assistant", "content": a})
    return messages

def clear_chat():
    global chat_history
    chat_history = []
    return []

# UI
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 📙 Multilingual PDF RAG System\nUpload a PDF and chat with it. Optionally enter a reference answer to evaluate with ROUGE-L.")

    with gr.Row():
        pdf_input = gr.File(label="Upload PDF", file_types=[".pdf"])
        process_btn = gr.Button("📄 Process PDF")
        status_output = gr.Textbox(label="Status", interactive=False)

    with gr.Row():
        chatbot = gr.Chatbot(label="Document Q&A", type="messages")

    with gr.Row():
        question_box = gr.Textbox(label="Your question", placeholder="Ask anything about the uploaded PDF...")
        reference_box = gr.Textbox(label="Reference Answer (Optional)", placeholder="Enter reference for ROUGE evaluation...")
    
    with gr.Row():
        ask_btn = gr.Button("🤖 Ask")
        clear_btn = gr.Button("🧹 Clear")

    process_btn.click(fn=process_pdf, inputs=pdf_input, outputs=status_output)
    ask_btn.click(fn=rag_query, inputs=[question_box, reference_box], outputs=chatbot)
    clear_btn.click(fn=clear_chat, inputs=[], outputs=chatbot)

if __name__ == "__main__":
    demo.launch()
