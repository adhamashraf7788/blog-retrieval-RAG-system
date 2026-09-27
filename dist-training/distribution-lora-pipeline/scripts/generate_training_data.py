
import requests
import pandas as pd

RAG_API_URL = "http://rag-service-host:PORT/retrieve"  # swap with actual endpoint


SPARK_OUTPUT_PATH = "s3://your-bucket/spark-output/cleaned_articles/"  # <-- fill in


def load_raw_articles():
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
