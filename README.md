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

5.  **Ollama Setup:**
    This project uses Ollama for local LLM inference. Ensure Ollama is installed and running.

    *   **Start Ollama Server:** Open a new terminal and run:
        ```bash
        ollama serve
        ```
        Keep this terminal open as long as you are running the `locsum` application.

    *   **Download LLM Model:** Download the `llama3` model (or your preferred model) using Ollama:
        ```bash
        ollama pull llama3
        ```

6.  **Configure Obsidian Vault Path:**
    Set the `OBSIDIAN_VAULT_PATH` environment variable to the absolute path of your Obsidian vault. This is crucial for the RAG system to load your notes.
    ```bash
    export OBSIDIAN_VAULT_PATH="/path/to/your/obsidian/vault"
    # Example: export OBSIDIAN_VAULT_PATH="/Users/youruser/Documents/Second brain"
    ```
    *Note: Replace `/path/to/your/obsidian/vault` with the actual path to your Obsidian notes. This variable needs to be set in every terminal session where you run the FastAPI application.*

7.  **Run the FastAPI Application:**
    Start the FastAPI application. It will automatically build the FAISS index from your Obsidian notes if it doesn't exist.
    ```bash
    ```bash
uvicorn src.app:app --reload --reload-exclude "faiss_index_A/" --reload-exclude "faiss_index_B/" --reload-exclude "index_state.json"
```
    ```

8.  **Access the Frontend:**
    Open your web browser and navigate to `http://127.0.0.1:8000/` to interact with the application.

## FastAPI Application Setup

A basic FastAPI application has been set up in `src/app.py` with a health check endpoint.

## RAG Core - Data Processing and Logic

### Document Loading

Implemented a function `load_documents_from_directory` in `src/rag_core.py` that uses `langchain_community.document_loaders.DirectoryLoader` to load markdown files from a specified directory. This function returns a list of `langchain_core.documents.Document` objects.

### Directory Structure Summary

To enable the RAG system to answer questions about the vault's structure (e.g., "How many files are in the 'Books' directory?"), a special summary document is generated on startup. This document contains a text-based representation of the entire directory tree, including folder names, file counts per folder, and the names of the files. This summary is then indexed along with the actual notes, giving the LLM the necessary context to answer questions about the vault's layout.

### Document Chunking

Implemented a function `chunk_documents` in `src/rag_core.py` that uses `langchain.text_splitter.RecursiveCharacterTextSplitter` to split loaded documents into smaller, manageable chunks. The default `chunk_size` is set to 1000 characters with a `chunk_overlap` of 200 characters. This initial choice is a common heuristic in RAG applications, aiming to balance the amount of context per chunk with the LLM's context window limitations. These parameters are configurable and can be fine-tuned for optimal performance.

### Embedding Model Initialization

Initialized the embedding model in `src/rag_core.py` using `langchain_community.embeddings.HuggingFaceEmbeddings`. The `all-mpnet-base-v2` model is used by default, configured to run on the most performant available device (CUDA > MPS > CPU). This model converts text into numerical vector embeddings, which are crucial for similarity search in the vector store.

**Why `all-mpnet-base-v2`?**

*   **Pros:**
    *   **High Performance:** It is a larger and more powerful model than `all-MiniLM-L6-v2`, offering better performance on a wide range of tasks. It is one of the top-performing sentence-transformer models.
    *   **Local Execution:** It can be run entirely offline, which is crucial for a privacy-focused personal second brain application.
*   **Cons:**
    *   **Resource Intensive:** Being a larger model, it requires more computational resources (RAM and VRAM) and is slower than smaller models like `all-MiniLM-L6-v2`.

### FAISS Vector Store Setup

Implemented a function `create_faiss_index` in `src/rag_core.py` that takes a list of documents and the initialized embedding model to create a FAISS vector store. FAISS (Facebook AI Similarity Search) is used for efficient similarity search of vector embeddings, enabling fast retrieval of relevant documents based on query embeddings.

### FAISS Index Persistence with Blue-Green Deployments

To ensure zero downtime, the application manages its FAISS vector index using a blue-green deployment strategy. This provides a seamless experience, even when your notes have changed and a full re-indexing is required.

#### How It Works

1.  **Two Index Directories:** The system maintains two separate index directories: `faiss_index_A` (blue) and `faiss_index_B` (green).
2.  **State Management:** A single state file, `index_state.json`, keeps track of which index is currently "live" (blue) and which is the candidate for the next switch (green).
3.  **Zero-Downtime Startup:** On startup, the application immediately loads the current blue index and becomes ready to serve queries. There is no waiting for indexing to complete.
4.  **Background Rebuilding:** After startup, a background task automatically checks if the notes or configuration have changed. If they have, it builds a completely new index in the inactive (green) directory without interrupting the live service.
5.  **User-Controlled Switchover:** Once the green index is built, it is marked as "ready." The user can then trigger a switchover via an API call, promoting the new green index to become the live blue index. This switch is instantaneous.

This approach guarantees that the application is always responsive and serving queries from a valid index, while new or updated indexes are built safely in the background.

#### Managing the Index

You can monitor and manage the index state using the following API endpoints:

*   **`GET /index-status`**: Check the current status of the blue and green indexes.
    *   **Response:**
        ```json
        {
          "blue_index": "A",
          "green_index": "B",
          "green_index_ready_for_swap": true
        }
        ```
*   **`POST /switch-index`**: Promotes the ready green index to be the new live blue index. This is the action that would be triggered by a "Use New Index" button in a UI.

#### Frontend Integration

The frontend (`src/static/index.html` and `src/static/script.js`) has been updated to provide a visual interface for this blue-green deployment:

*   **Status Bar:** A status bar at the top of the UI displays the current active index (e.g., "Active (A)") and indicates if a new index is ready for activation.
*   **"Switch to New Index" Button:** This button appears automatically when a new index has been successfully built in the background and is ready to be promoted. Clicking it triggers the `/switch-index` API call, making the new index live.

### Retrieval and Generation

Implemented functions in `src/rag_core.py` to set up the retrieval and generation components of the RAG system:

*   `get_retriever`: Creates a LangChain `VectorStoreRetriever` from the FAISS index, responsible for fetching relevant documents based on a query.
*   `get_llm`: Initializes and returns an Ollama LLM instance (defaulting to `llama3`), enabling local execution of large language models.
*   `get_rag_chain`: Constructs the complete RAG chain using LangChain Expression Language (LCEL). This chain orchestrates the retrieval of documents, formats them into a prompt, passes the prompt to the LLM, and parses the LLM's output to generate a coherent response.

## FastAPI Integration

Integrated the RAG components into the FastAPI application (`src/app.py`):

*   **`/query` Endpoint:** A new POST endpoint `/query` has been added to accept user queries. It utilizes the RAG chain to process the query and return a relevant response.
*   **Startup Logic:** On application startup, the FastAPI application now attempts to load the FAISS index from a predefined path (`./faiss_index`). If the index is not found, it automatically loads documents from a dummy directory (representing an Obsidian vault), chunks them, creates a new FAISS index, and saves it for future use. This ensures the RAG system is ready to serve queries upon startup.

## Running Tests

To run the unit tests for the FastAPI application and RAG core, ensure your virtual environment is activated and run:

```bash
export PYTHONPATH=$(pwd) && pytest tests/
```
