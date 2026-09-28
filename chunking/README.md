# Chunking

Multilingual news pipeline for the RAG system:

**WARC / CC-NEWS → extract & language filter (AR / EN / FR) → PostgreSQL → LangChain chunks → per-chunk summary / category / tags (Qwen) → JSON**


## Pipeline

```text
CC-NEWS .warc.gz
        │
        ▼
stream_to_db_parallel.py     # extract, filter language, store pages
        │
        ▼
PostgreSQL                   # websites / pages / metadata / content
        │
        ▼
langchain.py                 # load docs + split (chunk_size=800, overlap=100)
        │                    # each chunk has doc_id + chunk_index
        ▼
summarize_classify_final.py  # Groq: summary + category + tags PER CHUNK
        │                    # resume-safe, balanced AR/EN/FR
        ▼
chunk_analysis_results_*.json
````

Optional:

* `compare_keyword_methods.py` — YAKE vs KeyBERT vs TF-IDF on the **same** LLM-tagged chunks
* `final_evaluate_LLMs_models.py` — LLM output evaluation

## Layout

```text
chunking/
  stream_to_db_parallel.py     # WARC ingest (parallel)
  langchain.py                 # DB → LangChain documents + splitter
  summarize_classify_final.py  # per-chunk summary / classify / tags
  compare_keyword_methods.py   # keyword extractor benchmark
  final_evaluate_LLMs_models.py
  utils.py
  .env.example
  db/                           # Postgres schema + inserts
  extractors/                   # HTML / language extraction
  parsers/                      # WARC parsing
  Docs/                         # extra notes
```

## Setup

1. Copy the env file (never commit `.env`):

```bash
cp .env.example .env
```

2. Fill in `GROQ_API_KEY` and Postgres settings.

3. Install the Python dependencies used by this module (Groq, LangChain, psycopg2, trafilatura, …).

4. Create tables:

```bash
python db/db_handler.py
```

## How to run (order)

From this `chunking/` directory:

```bash
# 1) Ingest WARC into Postgres
python stream_to_db_parallel.py

# 2) Per-chunk summary + category + tags
python summarize_classify_final.py
```

`summarize_classify_final.py` will:

* load documents from Postgres
* split into chunks (`chunk_index` is set per document)
* pick a **balanced** batch (default 30 AR + 30 EN + 30 FR)
* skip already-`SUCCESS` chunks (resume after Groq daily quota)
* catch up lagging languages so totals stay even if a run stops mid-way

Re-run the same command the next day. It continues automatically.

Output (local, gitignored):

```text
chunk_analysis_results_last_version_multilingual_chunks.json
```

Each row includes: `doc_id`, `chunk_index`, `language`, `summary`, `category`, `tags`, `chunk_text_full`, `status`.

## Design notes (for other teams)

* **No article-level merge.** Each chunk is tagged on its own.
* Resume key is `(doc_id, chunk_index)` — not list index.
* Short / boilerplate chunks are skipped during selection so language counts stay equal.
* LLM tags (entities) are the preferred RAG metadata.
* Benchmark takeaway: YAKE is the fastest classic extractor. Use **Qwen tags as primary filters** and cleaned YAKE as a cheap fallback. Do not store raw TF-IDF terms as filters on this multilingual sample.


