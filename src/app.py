from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel # Import BaseModel for request body validation
from src.rag_core import (
    load_documents_from_directory,
    chunk_documents,
    initialize_embedding_model,
    create_faiss_index,
    save_faiss_index,
    load_faiss_index,
    get_retriever,
    get_llm,
    get_rag_chain,
    embedding_model # This is already initialized in rag_core.py
)
import os

app = FastAPI()

# Global variables to hold the FAISS index, retriever, LLM, and RAG chain
# These will be initialized on application startup.
faiss_index = None
retriever = None
llm = None
rag_chain = None

# Define the path for the FAISS index persistence.
# This path should be configurable in a real application (e.g., via environment variables).
FAISS_INDEX_PATH = "./faiss_index"

# Pydantic model for the query request body.
# This defines the expected structure of the JSON payload for the /query endpoint.
class QueryRequest(BaseModel):
    query: str

@app.on_event("startup")
async def startup_event():
    global faiss_index, retriever, llm, rag_chain

    # Initialize the embedding model (already done in rag_core.py, but good to be explicit)
    # embedding_model = initialize_embedding_model()

    # Check if the FAISS index already exists on disk.
    if os.path.exists(FAISS_INDEX_PATH):
        print(f"Loading FAISS index from {FAISS_INDEX_PATH}...")
        # Load the existing FAISS index.
        faiss_index = load_faiss_index(FAISS_INDEX_PATH, embedding_model)
    else:
        print("FAISS index not found. Creating a new one...")
        # For demonstration, we'll load documents from a dummy directory.
        # In a real application, this would be your Obsidian vault path.
        # Create a dummy directory and file for initial testing if it doesn't exist
        dummy_docs_path = "./dummy_obsidian_vault"
        os.makedirs(dummy_docs_path, exist_ok=True)
        with open(os.path.join(dummy_docs_path, "test_note.md"), "w") as f:
            f.write("This is a test note about the capital of France, which is Paris. "
                    "The Eiffel Tower is a famous landmark in Paris.")

        # Load, chunk, and create a new FAISS index.
        documents = load_documents_from_directory(dummy_docs_path)
        chunked_documents = chunk_documents(documents)
        faiss_index = create_faiss_index(chunked_documents, embedding_model)
        # Save the newly created index for future use.
        save_faiss_index(faiss_index, FAISS_INDEX_PATH)
        print(f"FAISS index created and saved to {FAISS_INDEX_PATH}")

    # Initialize the retriever from the FAISS index.
    retriever = get_retriever(faiss_index)

    # Initialize the LLM.
    llm = get_llm()

    # Initialize the RAG chain.
    rag_chain = get_rag_chain(retriever, llm)
    print("FastAPI application startup complete. RAG components initialized.")

@app.get("/health")
async def health_check():
    # Check if all RAG components are initialized.
    if faiss_index and retriever and llm and rag_chain:
        return {"status": "ok", "rag_initialized": True}
    else:
        return {"status": "ok", "rag_initialized": False}

@app.post("/query")
async def query_rag(request: QueryRequest): # Accept QueryRequest as the request body
    # Ensure RAG components are initialized before processing queries.
    if not rag_chain:
        raise HTTPException(status_code=503, detail="RAG components not initialized. Please wait for startup.")

    try:
        # Invoke the RAG chain with the user's query.
        response = rag_chain.invoke(request.query) # Access the query from the request object
        return JSONResponse(content={"response": response})
    except Exception as e:
        # Log the error for debugging purposes.
        print(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing query: {e}")
