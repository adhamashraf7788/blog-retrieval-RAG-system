DB_CONFIG = {
    "dbname": "multilingual_retrieval",
    "user": "postgres",
    "password": "HussenSabry#99",
    "host": "localhost",
    "port": 5432
}


EMBEDDING_MODEL = "BAAI/bge-m3"

SUPPORTED_LANGUAGES = [
    "ar",
    "en",
    "fr"
]

TOP_K = 5
FINAL_K = 3