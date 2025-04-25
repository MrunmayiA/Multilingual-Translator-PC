from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import evaluate
import re
from langdetect import detect
from llm.translation_utils import translate

# Load model
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🧠 Loading local model ({model_name}) on {device}...")

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32).to(device)

# Load ROUGE scorer
rouge = evaluate.load("rouge")

def normalize_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def generate_answer(query, context_chunks, reference=None, max_tokens=200):
    detected_lang = detect(query)
    print(f"🌐 Detected Question Language: {detected_lang}")

    # Fallback: Translate to English if not English
    if detected_lang != "en":
        print("🔁 Translating context and query to English...")
        query = translate(query, source_lang=detected_lang, target_lang="en")
        context_chunks = [translate(chunk, source_lang=detected_lang, target_lang="en") for chunk in context_chunks]

    # Join and trim context
    context = "\n\n".join(context_chunks)
    context = context[:1500].strip()

    if not context:
        return "The answer is not available in the document.", {}

    # Prompt template
    prompt = f"""You are a helpful assistant. Use ONLY the context below to answer the user's question.
If the answer is not in the context, say: "The answer is not available in the document."

### Context:
{context}

### Question:
{query}

### Answer:"""

    # Tokenize and generate
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_tokens,
        do_sample=False,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id
    )

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    answer = generated_text.split("### Answer:")[-1].strip()

    # Heuristic check
    if not answer or "not available" in answer.lower() or answer.lower() == query.lower():
        print("🛑 Vague or irrelevant answer detected.")
        return "The answer is not available in the document.", {}

    # Translate back if necessary
    if detected_lang != "en":
        print("🔁 Translating answer back to original language...")
        answer = translate(answer, source_lang="en", target_lang=detected_lang)

    print("🧠 Final Answer:", answer)

    # ROUGE score
    rouge_score = {}
    if reference:
        try:
            rouge_score = rouge.compute(predictions=[answer], references=[reference])
            print("📏 ROUGE Score:", rouge_score)
        except Exception as e:
            print("⚠️ ROUGE scoring failed:", e)

    return answer, rouge_score

def decompose_query(query):
    prompt = (
        "Split the user's complex question into 2 or 3 simple questions if needed. "
        "If it's already simple, return it as is. Use numbered format.\n\n"
        f"User Question: {query}\n\nSub-Questions:"
    )

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        do_sample=False,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id
    )

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("🧠 Decomposition Raw Output:\n", decoded)

    lines = decoded.split("\n")
    sub_questions = [line.lstrip("1234567890. ").strip() for line in lines if "?" in line]
    return sub_questions if sub_questions else [query.strip()]
