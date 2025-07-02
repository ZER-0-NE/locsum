from fastapi.testclient import TestClient
from src.app import app
from unittest.mock import patch, MagicMock

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    # The health check now also indicates if RAG components are initialized
    # During testing, they might not be fully initialized depending on startup events
    # So, we'll just check for the basic status: "ok"
    assert response.json()["status"] == "ok"

def test_query_rag():
    # Define a dummy query and expected response
    test_query = "What is the capital of France?"
    expected_response = "Paris is the capital of France."

    # Mock the rag_chain global variable in src.app
    with patch('src.app.rag_chain', new_callable=MagicMock) as mock_rag_chain:
        # Configure the mock's invoke method to return a predictable response
        mock_rag_chain.invoke.return_value = expected_response

        # Make a POST request to the /query endpoint
        response = client.post("/query", json={"query": test_query})

        # Assert the HTTP status code is 200 OK
        assert response.status_code == 200

        # Assert the response content matches the expected mocked response
        assert response.json() == {"response": expected_response}

        # Assert that the mock_rag_chain.invoke was called exactly once with the test_query
        mock_rag_chain.invoke.assert_called_once_with(test_query)
