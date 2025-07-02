from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
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
    embedding_model
)
import os
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the RAG components on startup
    
    # Define the path for the FAISS index persistence.
    FAISS_INDEX_PATH = os.environ.get("FAISS_INDEX_PATH", "./faiss_index")

    # Initialize the embedding model
    # embedding_model = initialize_embedding_model()

    # Check if the FAISS index already exists on disk.
    if os.path.exists(FAISS_INDEX_PATH):
        print(f"Loading FAISS index from {FAISS_INDEX_PATH}...")
        faiss_index = load_faiss_index(FAISS_INDEX_PATH, embedding_model)
    else:
        print("FAISS index not found. Creating a new one...")
        dummy_docs_path = "./dummy_obsidian_vault"
        os.makedirs(dummy_docs_path, exist_ok=True)
        with open(os.path.join(dummy_docs_path, "test_note.md"), "w") as f:
            f.write("This is a test note about the capital of France, which is Paris. "
                    "The Eiffel Tower is a famous landmark in Paris.")

        documents = load_documents_from_directory(dummy_docs_path)
        chunked_documents = chunk_documents(documents)
        faiss_index = create_faiss_index(chunked_documents, embedding_model)
        save_faiss_index(faiss_index, FAISS_INDEX_PATH)
        print(f"FAISS index created and saved to {FAISS_INDEX_PATH}")

    # Initialize the retriever, LLM, and RAG chain
    app.state.retriever = get_retriever(faiss_index)
    app.state.llm = get_llm()
    app.state.rag_chain = get_rag_chain(app.state.retriever, app.state.llm)
    
    print("FastAPI application startup complete. RAG components initialized.")
    
    yield
    
    # Clean up resources if needed on shutdown
    print("FastAPI application shutdown.")

app = FastAPI(lifespan=lifespan)

# Mount the static directory to serve frontend files
app.mount("/static", StaticFiles(directory="src/static"), name="static")

@app.get("/")
async def read_index():
    return FileResponse('src/static/index.html')

class QueryRequest(BaseModel):
    query: str

@app.get("/health")
async def health_check(request: Request):
    # Check if all RAG components are initialized in app.state
    if hasattr(request.app.state, 'rag_chain') and request.app.state.rag_chain:
        return {"status": "ok", "rag_initialized": True}
    else:
        return {"status": "ok", "rag_initialized": False}

@app.post("/query")
async def query_rag(request: Request, query_request: QueryRequest):
    # Ensure RAG components are initialized before processing queries.
    if not hasattr(request.app.state, 'rag_chain') or not request.app.state.rag_chain:
        raise HTTPException(status_code=503, detail="RAG components not initialized. Please wait for startup.")

    try:
        # Invoke the RAG chain with the user's query.
        response = request.app.state.rag_chain.invoke(query_request.query)
        return JSONResponse(content={"response": response})
    except Exception as e:
        # Log the error for debugging purposes.
        print(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing query: {e}")
