"""Centralised configuration for the Healix application.

All tuneable parameters are defined here as module-level constants.
Values can be overridden via environment variables where noted.
"""

import os

# ---------------------------------------------------------------------------
# Pinecone
# ---------------------------------------------------------------------------
PINECONE_INDEX_NAME: str = os.getenv("PINECONE_INDEX_NAME", "medbot")
PINECONE_DIMENSION: int = int(os.getenv("PINECONE_DIMENSION", "384"))
PINECONE_METRIC: str = os.getenv("PINECONE_METRIC", "cosine")
PINECONE_CLOUD: str = os.getenv("PINECONE_CLOUD", "aws")
PINECONE_REGION: str = os.getenv("PINECONE_REGION", "us-east-1")

# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------
EMBEDDING_MODEL_NAME: str = os.getenv(
    "EMBEDDING_MODEL_NAME",
    "sentence-transformers/all-MiniLM-L6-v2",
)

# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------
LLM_REPO_ID: str = os.getenv(
    "LLM_REPO_ID",
    "Qwen/Qwen2.5-7B-Instruct",
)
LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.4"))
LLM_MAX_NEW_TOKENS: int = int(os.getenv("LLM_MAX_NEW_TOKENS", "500"))

# ---------------------------------------------------------------------------
# Retriever
# ---------------------------------------------------------------------------
RETRIEVER_SEARCH_TYPE: str = os.getenv("RETRIEVER_SEARCH_TYPE", "similarity")
RETRIEVER_TOP_K: int = int(os.getenv("RETRIEVER_TOP_K", "3"))

# ---------------------------------------------------------------------------
# Text Splitter
# ---------------------------------------------------------------------------
CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "20"))

# ---------------------------------------------------------------------------
# Flask Server
# ---------------------------------------------------------------------------
HOST: str = os.getenv("FLASK_HOST", "0.0.0.0")
PORT: int = int(os.getenv("FLASK_PORT", "8080"))
DEBUG: bool = os.getenv("FLASK_DEBUG", "true").lower() == "true"
