"""
Run this ONCE, offline, before train.py — on whichever machine has network
access to the RAG service and can read Spark's output location (doesn't
need to be a VM with a GPU; this step does no training, just data prep).

Reads Spark's cleaned article output directly (Parquet — Spark's native
write format, no CSV conversion needed from the Spark side).

Produces cleaned_dialect_news.parquet with two columns: context, target.
"context" = the same retrieved-context prompt shape the RAG service builds
            at real inference time.
"target"  = the correct dialect-appropriate output for that article.

Fill in RAG_API_URL, SPARK_OUTPUT_PATH, and the request/response field
names once you have them — the values below are placeholders.
"""

import requests
import pandas as pd

RAG_API_URL = "http://rag-service-host:PORT/retrieve"  # ask your team for the real endpoint

# Wherever Spark's cleaned output actually lives — a local path, S3, HDFS, etc.
# pandas.read_parquet handles all of these transparently as long as the right
# filesystem libraries (s3fs, pyarrow, etc.) are installed.
SPARK_OUTPUT_PATH = "s3://your-bucket/spark-output/cleaned_articles/"  # <-- fill in


def load_raw_articles():
    # Reads however Spark partitioned its output (single file or a directory
    # of part-*.parquet files — pandas/pyarrow handle both transparently).
    df = pd.read_parquet(SPARK_OUTPUT_PATH)
    return df.to_dict(orient="records")


def build_training_row(article):
    response = requests.post(RAG_API_URL, json={"query": article["query"]})
    response.raise_for_status()
    retrieved_context = response.json()["context"]  # field name depends on their API's actual response shape

    prompt = f"Context: {retrieved_context}\n\nSummarize in dialect:"
    target = article["human_written_summary"]
    return {"context": prompt, "target": target}


def main():
    raw_articles = load_raw_articles()
    rows = [build_training_row(article) for article in raw_articles]
    df = pd.DataFrame(rows)
    df.to_parquet("cleaned_dialect_news.parquet", index=False)
    print(f"Wrote {len(df)} rows to cleaned_dialect_news.parquet")


if __name__ == "__main__":
    main()
