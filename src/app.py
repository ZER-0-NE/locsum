import os
import json
import shutil
import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from src.rag_core import (
    load_documents_from_directory,
    chunk_documents,
    create_faiss_index,
    save_faiss_index,
    load_faiss_index,
    get_retriever,
    get_llm,
    get_rag_chain,
    embedding_model,
    create_index_manifest,
    generate_directory_summary
)

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
INDEX_STATE_FILE = "./index_state.json"

class IndexState:
    def __init__(self):
        self.blue = "A"
        self.green = "B"
        self.green_ready = False
        self.load()

    def load(self):
        if os.path.exists(INDEX_STATE_FILE):
            with open(INDEX_STATE_FILE, 'r') as f:
                state = json.load(f)
                self.blue = state.get("blue", "A")
                self.green = state.get("green", "B")
                self.green_ready = state.get("green_ready", False)

    def save(self):
        with open(INDEX_STATE_FILE, 'w') as f:
            json.dump(self.__dict__, f, indent=4)

    def switch(self):
        self.blue, self.green = self.green, self.blue
        self.green_ready = False
        self.save()

    def get_blue_path(self):
        return f"./faiss_index_{self.blue}"

    def get_green_path(self):
        return f"./faiss_index_{self.green}"

index_state = IndexState()

async def rebuild_index_background(app: FastAPI):
    """A background task to rebuild the green index if it's stale."""
    await asyncio.sleep(5)  # Wait a bit for the server to be fully up
    print("BACKGROUND: Starting index check.")

    obsidian_vault_path = os.environ.get("OBSIDIAN_VAULT_PATH")
    if not obsidian_vault_path:
        print("BACKGROUND: OBSIDIAN_VAULT_PATH not set. Skipping rebuild.")
        return

    green_path = index_state.get_green_path()
    manifest_path = os.path.join(green_path, "manifest.json")

    all_documents = load_documents_from_directory(obsidian_vault_path)
    summary_doc = generate_directory_summary(obsidian_vault_path)
    all_documents.append(summary_doc)
    
    current_manifest = create_index_manifest(all_documents, CHUNK_SIZE, CHUNK_OVERLAP)

    rebuild_needed = True
    if os.path.exists(manifest_path):
        with open(manifest_path, 'r') as f:
            saved_manifest = json.load(f)
        if saved_manifest == current_manifest:
            rebuild_needed = False
            print("BACKGROUND: Green index is up to date.")
            if not index_state.green_ready:
                index_state.green_ready = True
                index_state.save()

    if rebuild_needed:
        print("BACKGROUND: Green index is stale or missing. Rebuilding...")
        if os.path.exists(green_path):
            shutil.rmtree(green_path)
        
        chunked_documents = chunk_documents(all_documents, CHUNK_SIZE, CHUNK_OVERLAP)
        new_index = create_faiss_index(chunked_documents, embedding_model)
        save_faiss_index(new_index, green_path)
        with open(manifest_path, 'w') as f:
            json.dump(current_manifest, f, indent=4)
        
        index_state.green_ready = True
        index_state.save()
        print(f"BACKGROUND: Successfully rebuilt green index at {green_path}")

def initialize_rag_chain(app: FastAPI, index_path: str):
    """Loads a FAISS index and initializes the RAG chain."""
    if not os.path.exists(index_path):
        raise FileNotFoundError(f"Cannot initialize RAG chain: Index path {index_path} not found.")
    
    faiss_index = load_faiss_index(index_path, embedding_model)
    app.state.retriever = get_retriever(faiss_index)
    app.state.llm = get_llm()
    app.state.rag_chain = get_rag_chain(app.state.retriever, app.state.llm)
    print(f"Successfully initialized RAG chain with index: {index_path}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 1. Load the blue (live) index immediately for zero-downtime startup
    blue_path = index_state.get_blue_path()
    print(f"Attempting to load blue index from: {blue_path}")
    if os.path.exists(blue_path):
        try:
            initialize_rag_chain(app, blue_path)
        except Exception as e:
            print(f"Could not load blue index: {e}. App will start without a RAG chain.")
            app.state.rag_chain = None
    else:
        print("No blue index found. App will start without a RAG chain.")
        app.state.rag_chain = None

    # 2. Start the background task to check and rebuild the green index
    asyncio.create_task(rebuild_index_background(app))
    
    yield
    
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
async def health_check():
    return {"status": "ok", "rag_initialized": hasattr(app.state, 'rag_chain') and app.state.rag_chain is not None}

@app.get("/index-status")
async def get_index_status():
    return {
        "blue_index": index_state.blue,
        "green_index": index_state.green,
        "green_index_ready_for_swap": index_state.green_ready
    }

@app.post("/switch-index")
async def switch_index():
    if not index_state.green_ready:
        raise HTTPException(status_code=409, detail="Green index is not ready to be switched.")

    try:
        green_path = index_state.get_green_path()
        print(f"Switching to green index: {green_path}")
        initialize_rag_chain(app, green_path)
        index_state.switch()
        return {"message": f"Successfully switched to index {index_state.blue}."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to switch index: {e}")

@app.post("/query")
async def query_rag(request: Request, query_request: QueryRequest):
    if not hasattr(request.app.state, 'rag_chain') or not request.app.state.rag_chain:
        raise HTTPException(status_code=503, detail="RAG chain not initialized. The index may be building.")

    try:
        response = request.app.state.rag_chain.invoke(query_request.query)
        return JSONResponse(content={"response": response})
    except Exception as e:
        print(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing query: {e}")