"""Healix — Pinecone index creation and document embedding script.

Run this script once to extract text from medical PDFs, chunk it,
compute embeddings, and store them in a Pinecone serverless index.

Usage::

    python store_index.py
"""

import logging
import os

from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

from src import config
from src.helper import download_hugging_face_embeddings, load_pdf_file, text_split

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

UPSERT_BATCH_SIZE = 100


def main():
    """Load PDFs, create a Pinecone index, and embed all document chunks."""
    pinecone_api_key = os.environ.get("PINECONE_API_KEY")
    if not pinecone_api_key:
        raise EnvironmentError("Missing required environment variable: PINECONE_API_KEY")

    logger.info("Loading PDF files from data/ ...")
    extracted_data = load_pdf_file(data="data/")

    logger.info("Splitting text into chunks (size=%d, overlap=%d) ...", config.CHUNK_SIZE, config.CHUNK_OVERLAP)
    text_chunks = text_split(extracted_data)
    logger.info("Created %d text chunks", len(text_chunks))

    logger.info("Downloading embedding model: %s", config.EMBEDDING_MODEL_NAME)
    embeddings = download_hugging_face_embeddings()

    logger.info("Connecting to Pinecone and creating index: %s", config.PINECONE_INDEX_NAME)
    pc = Pinecone(api_key=pinecone_api_key)

    pc.create_index(
        name=config.PINECONE_INDEX_NAME,
        dimension=config.PINECONE_DIMENSION,
        metric=config.PINECONE_METRIC,
        spec=ServerlessSpec(
            cloud=config.PINECONE_CLOUD,
            region=config.PINECONE_REGION,
        ),
    )

    index = pc.Index(config.PINECONE_INDEX_NAME)

    logger.info("Embedding documents and upserting into Pinecone ...")
    texts = [chunk.page_content for chunk in text_chunks]
    metadatas = [{"text": chunk.page_content, **chunk.metadata} for chunk in text_chunks]

    # Embed in batches and upsert
    for i in range(0, len(texts), UPSERT_BATCH_SIZE):
        batch_texts = texts[i : i + UPSERT_BATCH_SIZE]
        batch_meta = metadatas[i : i + UPSERT_BATCH_SIZE]
        batch_vectors = embeddings.embed_documents(batch_texts)

        vectors = [
            {"id": str(i + j), "values": vec, "metadata": meta}
            for j, (vec, meta) in enumerate(zip(batch_vectors, batch_meta))
        ]
        index.upsert(vectors=vectors)
        logger.info("  Upserted batch %d–%d of %d", i, min(i + UPSERT_BATCH_SIZE, len(texts)), len(texts))

    logger.info("Index '%s' created and populated successfully.", config.PINECONE_INDEX_NAME)


if __name__ == "__main__":
    main()