from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

model_name = "facebook/nllb-200-distilled-600M"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)  # <-- FIXED
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

def translate(text, source_lang="bn", target_lang="en"):
    print(f"🔁 Translating from {source_lang} to {target_lang}...")

    # Map to lang codes used by NLLB
    lang_map = {
        "en": "eng_Latn",
        "hi": "hin_Deva",
        "bn": "ben_Beng",
        "zh": "zho_Hans"
    }

    src = lang_map.get(source_lang, "eng_Latn")
    tgt = lang_map.get(target_lang, "eng_Latn")

    tokenizer.src_lang = src
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(device)

    generated_tokens = model.generate(
        **inputs,
        
        forced_bos_token_id=tokenizer.convert_tokens_to_ids(tgt),  # ✅ FIXED
        max_length=512
    )
    return tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
