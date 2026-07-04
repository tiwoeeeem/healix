"""Helper utilities for document loading, text splitting, and embeddings.

These functions are used by both the indexing pipeline (``store_index.py``)
and the RAG chain builder (``src.rag``).
"""

from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

from src import config


def load_pdf_file(data: str):
    """Load all PDF files from the given directory.

    Parameters
    ----------
    data : str
        Path to the directory containing PDF files.

    Returns
    -------
    list
        A list of LangChain ``Document`` objects extracted from the PDFs.
    """
    loader = DirectoryLoader(data, glob="*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    return documents


def text_split(extracted_data, chunk_size: int = config.CHUNK_SIZE, chunk_overlap: int = config.CHUNK_OVERLAP):
    """Split documents into smaller text chunks for embedding.

    Parameters
    ----------
    extracted_data : list
        List of LangChain ``Document`` objects to split.
    chunk_size : int, optional
        Maximum number of characters per chunk (default from config).
    chunk_overlap : int, optional
        Number of overlapping characters between consecutive chunks
        (default from config).

    Returns
    -------
    list
        A list of chunked ``Document`` objects.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    text_chunks = text_splitter.split_documents(extracted_data)
    return text_chunks


def download_hugging_face_embeddings(model_name: str = config.EMBEDDING_MODEL_NAME):
    """Download and initialise a HuggingFace sentence-transformer embedding model.

    Parameters
    ----------
    model_name : str, optional
        The HuggingFace model identifier (default from config).

    Returns
    -------
    HuggingFaceEmbeddings
        An embedding model instance ready for use with LangChain.
    """
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    return embeddings
