from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

model_name = "google/flan-t5-small"
device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)

def generate_answer(query, context_chunks, max_tokens=150):
    context = "\n\n".join(context_chunks)
    context = context[:1000]

    prompt = f"""Answer the question using only the context below.

Context:
{context}

Question: {query}
Answer:"""

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id
    )

    return tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

# Test it
if __name__ == "__main__":
    context = ["RAG stands for Retrieval-Augmented Generation, a technique to combine search with generation."]
    question = "What is RAG?"
    answer = generate_answer(question, context)
    print("\n✅ Generated Answer:\n", answer)
