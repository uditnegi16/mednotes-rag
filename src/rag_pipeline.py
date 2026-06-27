# ==========================================
# RAG Pipeline
# Connects Retriever + ChromaDB + LLM
# ==========================================

import chromadb
from sentence_transformers import SentenceTransformer

# -------------------------------
# Configuration
# -------------------------------

# Path where ChromaDB stores vectors
VECTOR_DB_PATH = "data/vectorstore"

# Collection name inside ChromaDB
COLLECTION_NAME = "medical_notes"

# Embedding model used during indexing
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Number of chunks to retrieve
TOP_K = 5

