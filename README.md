# locsum

This project aims to create a local, personal second brain using a Retrieval Augmented Generation (RAG) system. It will allow users to interact with their personal notes (initially from Obsidian) using natural language queries.

## Technologies Used

*   **Backend Framework:** Python with FastAPI
*   **LLM Orchestration:** LangChain
*   **Local LLM Provider:** Ollama (planned)
*   **Embedding Model:** Hugging Face Sentence-Transformers (planned)
*   **Vector Store:** FAISS (planned)
*   **Frontend:** Plain HTML, CSS, and JavaScript (planned)
*   **Web Search Integration:** Tavily Search API (planned)

## Project Setup and Environment Configuration

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/ZER-0-NE/locsum.git
    cd locsum
    ```

2.  **Create a Python Virtual Environment:**
    ```bash
    python3 -m venv .venv
    ```

3.  **Activate the Virtual Environment:**
    ```bash
    source .venv/bin/activate
    ```

4.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## FastAPI Application Setup

A basic FastAPI application has been set up in `src/app.py` with a health check endpoint.

## RAG Core - Data Processing and Logic

### Document Loading

Implemented a function `load_documents_from_directory` in `src/rag_core.py` that uses `langchain_community.document_loaders.DirectoryLoader` to load markdown files from a specified directory. This function returns a list of `langchain_core.documents.Document` objects.

### Document Chunking

Implemented a function `chunk_documents` in `src/rag_core.py` that uses `langchain.text_splitter.RecursiveCharacterTextSplitter` to split loaded documents into smaller, manageable chunks. The default `chunk_size` is set to 1000 characters with a `chunk_overlap` of 200 characters. This initial choice is a common heuristic in RAG applications, aiming to balance the amount of context per chunk with the LLM's context window limitations. These parameters are configurable and can be fine-tuned for optimal performance.

## Running Tests

To run the unit tests for the FastAPI application and RAG core, ensure your virtual environment is activated and run:

```bash
export PYTHONPATH=$(pwd) && pytest tests/
```
