import os
from dotenv import load_dotenv

load_dotenv()


LLM_MODEL = os.getenv(
    "LLM_MODEL"
)

HF_TOKEN = os.getenv(
    "HF_TOKEN"
)


EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "BAAI/bge-small-en-v1.5"
)


VECTOR_DB_PATH = os.getenv(
    "VECTOR_DB_PATH",
    "data/vectorstore"
)


COLLECTION_NAME = os.getenv(
    "COLLECTION_NAME",
    "mednotes"
)


RETRIEVAL_K = int(
    os.getenv(
        "RETRIEVAL_K",
        10
    )
)


RERANK_TOP_K = int(
    os.getenv(
        "RERANK_TOP_K",
        5
    )
)
BM25_WEIGHT = float(
    os.getenv(
        "BM25_WEIGHT",
        0.3
    )
)


VECTOR_WEIGHT = float(
    os.getenv(
        "VECTOR_WEIGHT",
        0.7
    )
)