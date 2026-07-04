"""RAG chain construction for the Healix application.

Provides a factory function that wires together embeddings, Pinecone (native SDK),
retriever, LLM, and prompt into a ready-to-invoke retrieval-augmented
generation chain.
"""

import logging
from typing import List

from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from pinecone import Pinecone

from src import config
from src.helper import download_hugging_face_embeddings
from src.prompt import system_prompt

logger = logging.getLogger(__name__)


class PineconeRetriever(BaseRetriever):
    """A LangChain-compatible retriever backed by the native Pinecone SDK.

    This avoids the broken ``langchain-pinecone`` package while keeping
    full compatibility with LangChain's chain abstractions.
    """

    index: object  # Pinecone Index instance
    embeddings: object  # HuggingFaceEmbeddings instance
    top_k: int = config.RETRIEVER_TOP_K

    class Config:
        arbitrary_types_allowed = True

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> List[Document]:
        """Embed the query and retrieve the top-k most similar documents."""
        query_vector = self.embeddings.embed_query(query)
        results = self.index.query(
            vector=query_vector,
            top_k=self.top_k,
            include_metadata=True,
        )
        documents = []
        for match in results.get("matches", []):
            metadata = match.get("metadata", {})
            text = metadata.pop("text", "")
            documents.append(Document(page_content=text, metadata=metadata))
        return documents


def build_rag_chain(huggingface_api_token: str):
    """Build and return a LangChain retrieval-augmented generation chain.

    Uses the native Pinecone SDK for vector retrieval, wrapped in a
    custom ``PineconeRetriever`` for LangChain compatibility.

    Parameters
    ----------
    huggingface_api_token : str
        API token used to authenticate with the HuggingFace Inference API.

    Returns
    -------
    chain
        A LangChain ``RetrievalChain`` that accepts ``{"input": "<query>"}``
        and returns a dict containing an ``"answer"`` key.
    """
    logger.info("Initialising embedding model: %s", config.EMBEDDING_MODEL_NAME)
    embeddings = download_hugging_face_embeddings(model_name=config.EMBEDDING_MODEL_NAME)

    logger.info("Connecting to Pinecone index: %s", config.PINECONE_INDEX_NAME)
    pc = Pinecone()
    index = pc.Index(config.PINECONE_INDEX_NAME)

    retriever = PineconeRetriever(
        index=index,
        embeddings=embeddings,
        top_k=config.RETRIEVER_TOP_K,
    )

    logger.info("Initialising LLM with repo_id: %s", config.LLM_REPO_ID)
    llm = HuggingFaceEndpoint(
        repo_id=config.LLM_REPO_ID,
        huggingfacehub_api_token=huggingface_api_token,
        temperature=config.LLM_TEMPERATURE,
        max_new_tokens=config.LLM_MAX_NEW_TOKENS,
    )
    chat_model = ChatHuggingFace(llm=llm)

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )

    question_answer_chain = create_stuff_documents_chain(chat_model, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    logger.info("RAG chain built successfully")
    return rag_chain
