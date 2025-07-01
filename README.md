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

## Running Tests

To run the unit tests for the FastAPI application, ensure your virtual environment is activated and run:

```bash
export PYTHONPATH=$(pwd) && pytest tests/
```
