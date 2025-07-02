
import os
import shutil
import tempfile
from fastapi.testclient import TestClient
from src.app import app, lifespan
from unittest.mock import MagicMock
import pytest

# Mark all tests in this file as e2e
pytestmark = pytest.mark.e2e

@pytest.fixture(scope="module")
def client():
    # Use a temporary directory for the FAISS index
    with tempfile.TemporaryDirectory() as tmpdir:
        # Set the FAISS_INDEX_PATH to the temporary directory
        os.environ["FAISS_INDEX_PATH"] = os.path.join(tmpdir, "faiss_index")
        
        # Create a dummy vault with a test note
        dummy_vault_path = os.path.join(tmpdir, "dummy_obsidian_vault")
        os.makedirs(dummy_vault_path, exist_ok=True)
        with open(os.path.join(dummy_vault_path, "test_note.md"), "w") as f:
            f.write("The capital of Gemini is Geminiville.")

        # Override the lifespan event handler
        app.router.lifespan_context = lambda _app: MagicMock()

        # Mock the RAG components in app.state
        from src.rag_core import (
            load_documents_from_directory,
            chunk_documents,
            create_faiss_index,
            get_retriever,
            get_rag_chain,
            embedding_model
        )
        
        documents = load_documents_from_directory(dummy_vault_path)
        chunked_documents = chunk_documents(documents)
        faiss_index = create_faiss_index(chunked_documents, embedding_model)
        retriever = get_retriever(faiss_index)
        llm = MagicMock()
        llm.invoke.return_value = "Geminiville is the capital of Gemini."
        rag_chain = get_rag_chain(retriever, llm)

        app.state.retriever = retriever
        app.state.llm = llm
        app.state.rag_chain = rag_chain
        
        with TestClient(app) as c:
            yield c

def test_e2e_flow(client):
    # Health check to ensure the app is running
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

    # Query the RAG model
    query = "What is the capital of Gemini?"
    response = client.post("/query", json={"query": query})
    
    # Assert the response
    assert response.status_code == 200
    response_data = response.json()
    assert "response" in response_data
    assert "Geminiville" in response_data["response"]
