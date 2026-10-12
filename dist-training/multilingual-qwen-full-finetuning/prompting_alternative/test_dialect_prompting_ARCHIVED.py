"""
Tests whether a well-written system prompt alone (no fine-tuning) gets a
model close enough on dialect handling that LoRA might not be necessary.

Setup:
    pip install groq
    $env:GROQ_API_KEY="your-key-here"     # PowerShell, from console.groq.com/keys
    # or: export GROQ_API_KEY="..."       # bash/Linux

Run:
    python test_qwen_prompting.py

Compares TWO models (Qwen — what your tech lead asked about — and ALLaM, a
model built specifically for Arabic/dialect handling) each against TWO
system prompts (weak generic vs. detailed dialect-aware), so you can see
both "does a good prompt help" and "does a dialect-specialized model help"
in one run.

Model availability on Groq changes — if either model ID below 404s, re-run
this to see your account's current list:
    python -c "from groq import Groq; import os; print([m.id for m in Groq(api_key=os.environ['GROQ_API_KEY']).models.list().data])"

Replace PLACEHOLDER_ARTICLES with real dialect samples once you have them.
"""

import os
import html
from groq import Groq

client = Groq(api_key=os.environ["GROQ_API_KEY"])

# Confirmed available on this account as of the last model-list check.
MODELS_TO_TEST = {
    "Qwen3.8-27b": "qwen/qwen3.8-27b",
    "ALLaM-2-7b (Arabic-specialized)": "allam-2-7b",
}

# ── Placeholder articles — swap these for real regional samples ──────
PLACEHOLDER_ARTICLES = [
    {
        "dialect": "Gulf (Khaleeji)",
        "text": "Oil prices rose three percent today after OPEC announced production cuts. "
                "Analysts expect further increases through the end of the quarter.",
    },
    {
        "dialect": "Egyptian",
        "text": "The Ministry of Education announced new exam dates for secondary school "
                "students, moving the schedule up by two weeks due to weather concerns.",
    },
    {
        "dialect": "Levantine",
        "text": "A new trade agreement between regional partners is expected to reduce "
                "tariffs on agricultural goods starting next month.",
    },
]

# ── Two system prompts to compare ─────────────────────────────────────
WEAK_SYSTEM_PROMPT = "Summarize the following news article."

STRONG_SYSTEM_PROMPT = """You are a news summarization assistant writing specifically for
a {dialect} Arabic-speaking audience. Summarize the article in 2-3 sentences,
written in natural {dialect} dialect — not Modern Standard Arabic, not English.
Use vocabulary, phrasing, and sentence structure a native {dialect} speaker would
actually use in everyday spoken or informal written Arabic. Keep the tone
conversational, the way a local news app would phrase it for this audience."""


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
    # NOTE: terminals (especially PowerShell/Windows Terminal) often render
    # right-to-left Arabic text incorrectly — words/lines can appear in the
    # wrong visual order even when the actual text is completely fine. Don't
    # judge Arabic quality from the terminal. This writes a proper HTML
    # report instead, which renders RTL correctly in any browser.
    html_parts = ["""<html><head><meta charset="utf-8">
    <style>
        body { font-family: Arial, sans-serif; max-width: 900px; margin: 40px auto; }
        .dialect-header { background: #222; color: white; padding: 10px; margin-top: 40px; }
        .model-header { background: #555; color: white; padding: 8px; margin-top: 20px; }
        .label { font-weight: bold; color: #555; margin-top: 12px; }
        .arabic { direction: rtl; text-align: right; font-size: 18px; line-height: 1.8;
                   background: #f5f5f5; padding: 12px; border-radius: 6px; }
        .english { background: #f5f5f5; padding: 12px; border-radius: 6px; }
    </style></head><body>
    <h1>Qwen vs. ALLaM — Prompt Comparison Results</h1>"""]

    for article in PLACEHOLDER_ARTICLES:
        dialect = article["dialect"]
        text = article["text"]
        strong_prompt = STRONG_SYSTEM_PROMPT.format(dialect=dialect)

        print(f"Running: {dialect}...")
        html_parts.append(f'<div class="dialect-header"><b>DIALECT:</b> {html.escape(dialect)}<br>'
                           f'<b>ARTICLE:</b> {html.escape(text)}</div>')

        for model_label, model_id in MODELS_TO_TEST.items():
            html_parts.append(f'<div class="model-header">MODEL: {html.escape(model_label)}</div>')

            weak_output = ask(model_id, WEAK_SYSTEM_PROMPT, text)
            html_parts.append(f'<div class="label">Weak prompt:</div>'
                               f'<div class="english">{html.escape(weak_output)}</div>')

            strong_output = ask(model_id, strong_prompt, text)
            html_parts.append(f'<div class="label">Strong prompt (dialect-aware):</div>'
                               f'<div class="arabic">{html.escape(strong_output)}</div>')

    html_parts.append("</body></html>")

    output_path = "comparison_results.html"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(html_parts))

    print(f"\nDone. Open {output_path} in a browser to read the Arabic correctly")
    print("(do NOT judge Arabic quality from this terminal — PowerShell often")
    print("renders right-to-left text in the wrong order even when it's fine).")


if __name__ == "__main__":
    main()
