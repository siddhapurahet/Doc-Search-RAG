"""Vector store for document embedding and retreival."""

import json
import time
from typing import List

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from src.config.config import Config


class VectorStore:
    """Vector store for document embedding and retreival"""

    def __init__(self):
        self.embedding = Config.get_embeddings()
        self.vectorstore = None
        self.retriever = None

    def create_retriever(self, documents: List[Document]):
        """Create a new retriever
        Args:
            documents: List of documents to embed and store.
        """
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
