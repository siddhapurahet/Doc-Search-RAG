from typing import List
# from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from typing import List, Union
from pathlib import Path
from langchain_community.document_loaders import (
    WebBaseLoader,
    PyPDFLoader,
    TextLoader,
    PyPDFDirectoryLoader,
)

class DocumentProcessor:
    """Handle document loading and processing."""

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        """
        Initialize the document processor.

        Args:
            chunk_size: Size of the text chunks.
            chunk_overlap: The overlap between the chunks.
        """

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )

    def load_from_url(self, url: str) -> List[Document]:
        """Load documents from a URL."""
        loader = WebBaseLoader(url)
        documents = loader.load()
        return documents

    def load_from_pdf_dir(self, directory: Union[str, Path]) -> List[Document]:
        """Load documents from a PDF file."""
        loader = PyPDFDirectoryLoader(str(directory))
        documents = loader.load()
        return documents

    def load_from_text(self, file_path: Union[str, Path]) -> List[Document]:
        """Load documents from a text file."""
        loader = TextLoader(str(file_path), encoding="utf-8")
        documents = loader.load()
        return documents

    def load_from_pdf(self, file_path: Union[str, Path]) -> List[Document]:
        """Load documents from a PDF file."""
        loader = PyPDFDirectoryLoader(str("data"))
        documents = loader.load()
        return documents

    def load_documents(self, sources: List[str]) -> List[Document]:
        """Load documents from URL's, file or directory.
        Args:
            sources: List of URLs, file paths or directory paths.
        Returns:
            List of loaded documents.
        """

        docs: List[Document] = []

        for source in sources:
            if source.startswith("http://") or source.startswith("https://"):
                docs.extend(self.load_from_url(source))
                continue

            path = Path(source)
            if path.is_dir():
                docs.extend(self.load_from_pdf_dir(path))
            elif path.suffix.lower() == ".txt":
                docs.extend(self.load_from_text(path))
            elif path.suffix.lower() == ".pdf":
                docs.extend(self.load_from_pdf(path))
            else:
                raise ValueError(f"Unsupported file type: {source}")
        return docs

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into chunks.
        Args:
            documents: List of documents to split.
        Returns:
            List of split documents.
        """
        return self.splitter.split_documents(documents)

    def process_documents(self, documents: List[str]) -> List[Document]:
        """Process documents (urls, files or directories).
        Args:
            documents: List of documents to process.
        Returns:
            List of processed documents.
        """
        docs = self.load_documents(documents)
        return self.split_documents(docs)