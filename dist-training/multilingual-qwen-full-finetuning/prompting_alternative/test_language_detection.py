"""
Tests whether an LLM can reliably (1) detect which language an article is
written in — Arabic, English, or French — and (2) respond/summarize in
that SAME language (standard form, not dialect), using prompting alone.

This supersedes the earlier dialect-specific test (test_qwen_prompting.py).
Scope changed: no more per-dialect handling, just detect + respond in
Arabic (standard/MSA), English, or French.

Setup:
    pip install groq
    $env:GROQ_API_KEY="your-key-here"     # PowerShell
    # or: export GROQ_API_KEY="..."       # bash/Linux

Run:
    python test_language_detection.py

Tests 3 models: Qwen (general multilingual), GPT-OSS-120b (general
multilingual), and ALLaM (Arabic-specialized — included to see whether an
Arabic-focused model is still worth it for standard Arabic, or whether it
falls short on English/French since it's not built for those).

Model availability on Groq changes — re-check your account's list if any
model ID below 404s:
    python -c "from groq import Groq; import os; print([m.id for m in Groq(api_key=os.environ['GROQ_API_KEY']).models.list().data])"

Replace PLACEHOLDER_ARTICLES with real samples once you have them.
"""

import os
import html
from groq import Groq

client = Groq(api_key=os.environ["GROQ_API_KEY"])

MODELS_TO_TEST = {
    "Qwen3.8-27b": "qwen/qwen3.8-27b",
    "GPT-OSS-120b": "openai/gpt-oss-120b",
    "ALLaM-2-7b (Arabic-specialized)": "allam-2-7b",
}

# ── Placeholder articles — one per language, swap for real samples ───
PLACEHOLDER_ARTICLES = [
    {
        "language": "Arabic (MSA)",
        "text": "أعلنت وزارة التجارة اليوم عن اتفاقية جديدة لخفض الرسوم الجمركية "
                "على المنتجات الزراعية بين الدول المشاركة، على أن يبدأ التنفيذ "
                "الشهر المقبل.",
    },
    {
        "language": "English",
        "text": "The Ministry of Trade announced a new agreement today to reduce "
                "tariffs on agricultural products between participating countries, "
                "with implementation set to begin next month.",
    },
    {
        "language": "French",
        "text": "Le ministère du Commerce a annoncé aujourd'hui un nouvel accord "
                "visant a reduire les droits de douane sur les produits agricoles "
                "entre les pays participants, dont la mise en oeuvre debutera le "
                "mois prochain.",
    },
]

# ── Two system prompts to compare ─────────────────────────────────────
WEAK_SYSTEM_PROMPT = "Summarize the following news article."

STRONG_SYSTEM_PROMPT = """You are a news summarization assistant. First, detect which
language the article is written in — it will be Arabic, English, or French.
Then write a 2-3 sentence summary in that SAME language. If the article is in
Arabic, respond in formal Modern Standard Arabic (not a regional dialect).
Do not translate the article into a different language under any circumstances —
the summary language must match the article's original language exactly."""


def ask(model_id, system_prompt, article_text):
    response = client.chat.completions.create(
        model=model_id,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": article_text},
        ],
        temperature=0.7,
        max_tokens=300,
    )
    return response.choices[0].message.content


def main():
    # Same RTL caveat as before — judge Arabic output in the HTML file,
    # never in a PowerShell/terminal window.
    html_parts = ["""<html><head><meta charset="utf-8">
    <style>
        body { font-family: Arial, sans-serif; max-width: 900px; margin: 40px auto; }
        .lang-header { background: #222; color: white; padding: 10px; margin-top: 40px; }
        .model-header { background: #555; color: white; padding: 8px; margin-top: 20px; }
        .label { font-weight: bold; color: #555; margin-top: 12px; }
        .arabic { direction: rtl; text-align: right; font-size: 18px; line-height: 1.8;
                   background: #f5f5f5; padding: 12px; border-radius: 6px; }
        .other { background: #f5f5f5; padding: 12px; border-radius: 6px; }
    </style></head><body>
    <h1>Language Detection + Same-Language Response — Comparison Results</h1>"""]

    for article in PLACEHOLDER_ARTICLES:
        language = article["language"]
        text = article["text"]

        print(f"Running: {language}...")
        html_parts.append(f'<div class="lang-header"><b>SOURCE LANGUAGE:</b> {html.escape(language)}<br>'
                           f'<b>ARTICLE:</b> {html.escape(text)}</div>')

        for model_label, model_id in MODELS_TO_TEST.items():
            html_parts.append(f'<div class="model-header">MODEL: {html.escape(model_label)}</div>')

            weak_output = ask(model_id, WEAK_SYSTEM_PROMPT, text)
            css_class = "arabic" if language.startswith("Arabic") else "other"
            html_parts.append(f'<div class="label">Weak prompt (no language instruction):</div>'
                               f'<div class="{css_class}">{html.escape(weak_output)}</div>')

            strong_output = ask(model_id, STRONG_SYSTEM_PROMPT, text)
            html_parts.append(f'<div class="label">Strong prompt (detect + match language):</div>'
                               f'<div class="{css_class}">{html.escape(strong_output)}</div>')

    html_parts.append("</body></html>")

    output_path = "language_detection_results.html"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(html_parts))

    print(f"\nDone. Open {output_path} in a browser.")
    print("Check: did EVERY output stay in the SAME language as its source article?")
    print("A failure here looks like: Arabic article -> English summary (or vice versa),")
    print("or Arabic summary that reads as dialect instead of formal MSA.")


if __name__ == "__main__":
    main()
