"""
Run this ONCE, offline, before train.py — on whichever machine has raw
articles + network access to the RAG service (doesn't need to be a VM
with a GPU; this step does no training, just data prep).

Produces cleaned_dialect_news.csv with two columns: context, target.
"context" = the same retrieved-context prompt shape the RAG service builds
            at real inference time.
"target"  = the correct dialect-appropriate output for that article.

Fill in RAG_API_URL and the request/response field names once you have
them from your team — the values below are placeholders.
"""

import requests
import pandas as pd

RAG_API_URL = "http://rag-service-host:PORT/retrieve"  # ask your team for the real endpoint

# raw_articles should be a list of your own dicts, e.g.:
# [{"query": "...", "human_written_summary": "..."}, ...]
# Replace this with however you're actually loading your region's raw articles.
raw_articles = []  # <-- fill in


def build_training_row(article):
    response = requests.post(RAG_API_URL, json={"query": article["query"]})
    response.raise_for_status()
    retrieved_context = response.json()["context"]  # field name depends on their API's actual response shape

    prompt = f"Context: {retrieved_context}\n\nSummarize in dialect:"
    target = article["human_written_summary"]
    return {"context": prompt, "target": target}


def main():
    rows = [build_training_row(article) for article in raw_articles]
    df = pd.DataFrame(rows)
    df.to_csv("cleaned_dialect_news.csv", index=False)
    print(f"Wrote {len(df)} rows to cleaned_dialect_news.csv")


if __name__ == "__main__":
    main()
