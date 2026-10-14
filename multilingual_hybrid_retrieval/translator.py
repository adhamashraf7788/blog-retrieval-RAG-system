# User Query
#     ↓
# Language Detection
#     ↓
# NLLB-200 Translation
#     ↓
# English Query

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch


MODEL_NAME = "facebook/nllb-200-distilled-600M"


device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Loading translation model on {device}...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForSeq2SeqLM.from_pretrained(
    MODEL_NAME
).to(device)


def translate_to_english(text: str, source_language: str) -> str:

    language_codes = {
        "ar": "arb_Arab",
        "en": "eng_Latn",
        "fr": "fra_Latn"
    }

    source_code = language_codes[source_language]

    tokenizer.src_lang = source_code

    inputs = tokenizer(
        text,
        return_tensors="pt"
    ).to(device)

    generated_tokens = model.generate(
        **inputs,
        forced_bos_token_id=tokenizer.convert_tokens_to_ids("eng_Latn"),
        max_length=512
    )

    translated_text = tokenizer.batch_decode(
        generated_tokens,
        skip_special_tokens=True
    )[0]

    return translated_text


#if __name__ == "__main__":

#    query = "كيف تساعد قواعد البيانات المتجهة في تحسين البحث الدلالي؟"

#    translated = translate_to_english(
#        query,
#        source_language="ar"
#    )

#    print("\nOriginal:")
#    print(query)

#    print("\nTranslated:")
#    print(translated)
if __name__ == "__main__":

    queries = [
        ("ما هو semantic search في artificial intelligence؟", "en"),
        ("كيف تعمل vector databases في machine learning؟", "ar"),
        ("ما هو RAG system وكيف يعمل؟", "ar"),
        ("What is البحث الدلالي في artificial intelligence؟", "en"),
    ]

    for query, source_language in queries:

        translated = translate_to_english(
            query,
            source_language=source_language
        )

        print("\n" + "=" * 60)
        print("Original:")
        print(query)

        print("\nSource Language:")
        print(source_language)

        print("\nTranslated:")
        print(translated)