"""RAG state definition for LangGraph."""

from typing import List
from langchain_core.documents import Document
from pydantic import BaseModel

class RAGState(BaseModel):
    """State object for RAG workflow"""
    question: str
    retreived_docs: List[Document] = []
    answer: str = ""
