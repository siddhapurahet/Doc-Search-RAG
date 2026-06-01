# Doc Search RAG
Doc Search RAG is a Streamlit-based Retrieval-Augmented Generation application that ingests documents, indexes them into a vector store, retrieves relevant context for a user query, and generates an answer using an LLM.

The system is structured around a RAG pipeline:

<img width="1349" height="180" alt="Screenshot 2026-06-01 at 10 51 16 AM" src="https://github.com/user-attachments/assets/673a4453-2ae2-4875-a009-a5bce5b62760" />

### Architecture
The project is organized into modules, each responsible for one part of the RAG workflow.

**src/config**

Responsibilities: <br>
• Loads environment variables from .env <br>
• Configures the Gemini LLM provider <br>
• Configures embedding behavior <br>
• Defines chunking parameters <br>
• Provides default document URLs <br>
• The application expects the following environment variable: RAG_API_KEY=your_google_api_key_here <br>

**src/document_ingestion**

Responsibilities: <br>
• Load documents from URLs, text files, PDFs, or directories <br>
• Convert raw sources into LangChain Document objects <br>
• Split large documents into overlapping chunks using RecursiveCharacterTextSplitter <br>

**src/vector_store**

Responsibilities: <br>
• Initializes the embedding model <br>
• Converts document chunks into embeddings <br>
• Stores embeddings in a FAISS vector index <br>
• Exposes a retriever interface for similarity search <br>
At startup, all processed document chunks are embedded and stored in FAISS. During query time, the retriever searches this index for chunks most relevant to the user’s question. <br>

**src/state**

The main state object is RAGState, which contains: <br>
question        -> the user query <br>
retreived_docs  -> documents retrieved from the vector store <br>
answer          -> final generated response <br>
This state object acts as the contract between workflow nodes. <br>

**src/nodes**

The current workflow has two main nodes: <br>
• Retrieve documents <br>
• Takes the user question <br>
• Calls the vector store retriever <br>
• Adds relevant document chunks to the graph state <br>

Generate answer <br>
• Combines retrieved chunks into context <br>
• Builds a prompt using the context and question <br>
• Sends the prompt to the configured LLM <br>
• Stores the generated answer in state <br>

**src/graph_builder**

The graph builder wires together: <br>
• The RAGState <br>
• The retrieval node <br>
• The answer generation node <br>
• The retriever from the vector store <br>
• The LLM from configuration <br>
This keeps orchestration separate from business logic. <br>

**streamlit_app.py**

Responsibilities: <br>
• Initialize the RAG system once using Streamlit caching <br>
• Load and process documents <br>
• Build the vector store <br>
• Compile the LangGraph workflow <br>
• Accept user questions <br>
• Display generated answers and retrieved source documents <br>

**Running the Application**

- Install dependencies: uv sync <br>
- Create a local .env file: cp .env.example .env <br>
- Add your API key: RAG_API_KEY=your_google_gemini_api_key <br>
- Run the Streamlit app: python3 -m streamlit run streamlit_app.py <br>

# Output screenshots
<img width="663" height="1019" alt="ss1" src="https://github.com/user-attachments/assets/c5a92e14-192e-4483-9cd4-c72fc98401a3" />
<img width="519" height="1071" alt="ss2" src="https://github.com/user-attachments/assets/19555401-2483-45bd-8708-f4572cc657dc" />




