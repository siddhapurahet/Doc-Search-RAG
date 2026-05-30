"""Vector store for document embedding and retreival."""

import json
import time
from typing import List

from langchain_community.vectorstores import FAISS
from langchain.schema import Document

from src.config.config import Config


class VectorStore:
    """Vector store for document embedding and retreival"""

    def __init__(self):
        self.embedding = Config.get_embeddings()
        self.vectorstore = None
        self.retriever = None
        # #region agent log
        try:
            with open("/Users/harshsiddhapura/Downloads/Het/Doc-Search-RAG/.cursor/debug-980812.log", "a") as _f:
                _f.write(json.dumps({"sessionId": "980812", "runId": "post-fix", "hypothesisId": "A", "location": "vector_store.py:__init__", "message": "Embedding provider initialized", "data": {"provider": Config.EMBEDDING_PROVIDER, "model": Config.LOCAL_EMBEDDING_MODEL if Config.EMBEDDING_PROVIDER == "local" else Config.EMBEDDING_MODEL}, "timestamp": int(time.time() * 1000)}) + "\n")
        except Exception:
            pass
        # #endregion

    def create_retriever(self, documents: List[Document]):
        """Create a new retriever
        Args:
            documents: List of documents to embed and store.
        """
        # #region agent log
        try:
            with open("/Users/harshsiddhapura/Downloads/Het/Doc-Search-RAG/.cursor/debug-980812.log", "a") as _f:
                _f.write(json.dumps({"sessionId": "980812", "runId": "post-fix", "hypothesisId": "B", "location": "vector_store.py:create_retriever", "message": "Creating FAISS index", "data": {"doc_count": len(documents), "embedding_provider": Config.EMBEDDING_PROVIDER}, "timestamp": int(time.time() * 1000)}) + "\n")
        except Exception:
            pass
        # #endregion
        self.vectorstore = FAISS.from_documents(documents, self.embedding)
        self.retriever = self.vectorstore.as_retriever()

    def get_retriever(self):
        """Get the retriever object
        Returns:
            Retriever object.
        """
        if self.retriever is None:
            raise ValueError("Vector store not initialized. Call create_retriever first.")
        return self.retriever

    def retrieve(self, query: str, k: int = 4) -> List[Document]:
        """
        Retrieve relevant documents for a query from the vector store
        Args:
            query: Query to retrieve documents from.
            k: Number of documents to retrieve.
        Returns:
            List of relevant documents.
        """
        if self.retriever is None:
            raise ValueError("Vector store not initialized. Call create_retriever first.")
        return self.retriever.invoke(query)
