"""Healix — Flask application entry point.

A medical diagnosis chatbot powered by retrieval-augmented generation (RAG)
using Mistral LLM, LangChain, and Pinecone.
"""

import logging
import os

from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request

from src import config
from src.rag import build_rag_chain

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def create_app():
    """Application factory — creates and configures the Flask app.

    Returns
    -------
    Flask
        A fully configured Flask application with the RAG chain initialised.
    """
    app = Flask(__name__)

    # ── Validate required environment variables ──────────────────────────
    pinecone_api_key = os.environ.get("PINECONE_API_KEY")
    huggingface_api_key = os.environ.get("HUGGINGFACEHUB_API_KEY")

    if not pinecone_api_key:
        raise EnvironmentError("Missing required environment variable: PINECONE_API_KEY")
    if not huggingface_api_key:
        raise EnvironmentError("Missing required environment variable: HUGGINGFACEHUB_API_KEY")

    # ── Build the RAG chain ──────────────────────────────────────────────
    rag_chain = build_rag_chain(huggingface_api_token=huggingface_api_key)

    # ── Routes ───────────────────────────────────────────────────────────
    @app.route("/")
    def index():
        """Serve the chat interface."""
        return render_template("chat.html")

    @app.route("/get", methods=["GET", "POST"])
    def chat():
        """Handle a user query and return the RAG-generated answer."""
        user_message = request.form.get("msg", "").strip()

        if not user_message:
            logger.warning("Received empty message from client")
            return jsonify({"error": "Message cannot be empty"}), 400

        logger.info("User query: %s", user_message)

        try:
            response = rag_chain.invoke({"input": user_message})
            answer = response["answer"]
            logger.info("Response: %s", answer)
            return str(answer)
        except Exception:
            logger.exception("Error processing query")
            return jsonify({"error": "An error occurred while processing your request"}), 500

    return app


if __name__ == "__main__":
    app = create_app()
    app.run(
        host=config.HOST,
        port=config.PORT,
        debug=config.DEBUG,
        use_reloader=False,
        threaded=False,
    )