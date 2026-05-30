"""RAG state definition for LangGraph."""

from typing import List
from pydantic import BaseModel
from langchain.schema import Document

class RAGState(BaseModel):
    """State object for RAG workflow"""
    question: str
    retreived_docs: List[Document] = []
    answer: str = ""
