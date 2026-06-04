"""Configuration module for Agentic RAG System"""

import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

# Load environment variables
load_dotenv()

class Config:
    """Configuration class for RAG system"""

    # API Keys
    RAG_API_KEY = os.getenv("RAG_API_KEY")

    # Model configuration (provider:model_id for init_chat_model)
    LLM_MODEL_PROVIDER = "google_genai"
    LLM_MODEL = "gemini-2.5-flash"
    EMBEDDING_MODEL = "gemini-embedding-001"
    EMBEDDING_PROVIDER = "google"  # "local" avoids Google free-tier embed rate limits
    LOCAL_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

    #Document Processing
    CHUNK_SIZE = 1500
    CHUNK_OVERLAP = 100

    # Default url's
    DEFAULT_URLS = [
        "https://lilianweng.github.io/posts/2023-06-23-agent/",
        "https://lilianweng.github.io/posts/2024-04-12-diffusion-video/"
    ]
    DEFAULT_SOURCES = DEFAULT_URLS[:1]

    @classmethod
    def configure_api_key(cls):
        """Set Google API key env var for LangChain integrations."""
        if cls.RAG_API_KEY:
            os.environ["GOOGLE_API_KEY"] = cls.RAG_API_KEY

    @classmethod
    def get_embeddings(cls):
        """Return embedding model (local by default to avoid API quotas)."""
        if cls.EMBEDDING_PROVIDER == "google":
            cls.configure_api_key()
            from langchain_google_genai import GoogleGenerativeAIEmbeddings
            return GoogleGenerativeAIEmbeddings(
                model=cls.EMBEDDING_MODEL,
                google_api_key=cls.RAG_API_KEY,
            )
        from langchain_community.embeddings import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings(model_name=cls.LOCAL_EMBEDDING_MODEL)

    @classmethod
    def get_llm(cls):
        """Inititalize and return the LLM model"""
        cls.configure_api_key()
        return init_chat_model(cls.LLM_MODEL, model_provider=cls.LLM_MODEL_PROVIDER)