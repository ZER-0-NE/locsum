import os
import shutil
import tempfile
from src.rag_core import load_documents_from_directory, chunk_documents, initialize_embedding_model, create_faiss_index, save_faiss_index, load_faiss_index
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

def test_load_documents_from_directory():
    # Define a temporary directory for testing.
    test_dir = "./temp_test_docs"
    # Define paths for dummy markdown files within the temporary directory.
    os.makedirs(test_dir, exist_ok=True)
    file1_path = os.path.join(test_dir, "doc1.md")
    file2_path = os.path.join(test_dir, "subdir", "doc2.md")

    # Create dummy markdown files with content.
    with open(file1_path, "w") as f:
        f.write("This is document 1.")
    os.makedirs(os.path.dirname(file2_path), exist_ok=True)
    with open(file2_path, "w") as f:
        f.write("This is document 2.")

    try:
        # Call the function to load documents from the temporary directory.
        documents = load_documents_from_directory(test_dir)

        # Assert that two documents were loaded.
        assert len(documents) == 2

        # Assert the content of the loaded documents.
        # The order of documents might not be guaranteed, so check both.
        contents = [doc.page_content for doc in documents]
        assert "This is document 1." in contents
        assert "This is document 2." in contents

        # Assert that the metadata contains the correct source paths.
        # Normalize paths for comparison as DirectoryLoader might return relative paths.
        sources = [os.path.normpath(doc.metadata["source"]) for doc in documents]
        expected_sources = [os.path.normpath(file1_path), os.path.normpath(file2_path)]

        # Check if all expected sources are present in the loaded document sources.
        for expected_source in expected_sources:
            assert expected_source in sources

    finally:
        # Clean up the temporary directory and its contents.
        shutil.rmtree(test_dir)

def test_load_documents_from_nonexistent_directory():
    # Define a path to a directory that does not exist.
    non_existent_dir = "./non_existent_dir"
    try:
        # Expect a FileNotFoundError when trying to load from a non-existent directory.
        load_documents_from_directory(non_existent_dir)
        # If no error is raised, fail the test.
        assert False, "FileNotFoundError was not raised for a non-existent directory."
    except FileNotFoundError as e:
        # Assert that the error message is as expected.
        assert f"Directory not found: {non_existent_dir}" in str(e)

def test_chunk_documents():
    # Create a dummy long document for testing chunking.
    long_text = "This is a very long document that needs to be split into smaller chunks. " \
                "Each chunk should have a specific size and an overlap with the next chunk. " \
                "This helps in maintaining context across the chunks when performing retrieval. " \
                "The RecursiveCharacterTextSplitter is designed to intelligently split text " \
                "while trying to keep meaningful units together, like sentences or paragraphs. " \
                "This is the final part of the document."
    
    # Create a LangChain Document object from the dummy text.
    # Metadata can be added here if needed, but for chunking, page_content is sufficient.
    dummy_document = Document(page_content=long_text, metadata={"source": "dummy_source.txt"})

    # Define chunking parameters.
    chunk_size = 100
    chunk_overlap = 20

    # Call the chunking function with the dummy document.
    chunked_documents = chunk_documents([dummy_document], chunk_size, chunk_overlap)

    # Assert that the document has been split into multiple chunks.
    # The exact number of chunks depends on the text and chunking parameters.
    assert len(chunked_documents) > 1

    # Assert that each chunk's content length is within the expected range (considering overlap).
    for chunk in chunked_documents:
        assert len(chunk.page_content) <= chunk_size
        # Also check that metadata is preserved.
        assert chunk.metadata["source"] == "dummy_source.txt"

    # Removed strict overlap assertion as RecursiveCharacterTextSplitter does not guarantee exact substring overlap.
    # The primary goal is to ensure documents are chunked and metadata is preserved.

def test_initialize_embedding_model():
    # Call the function to initialize the embedding model.
    embeddings = initialize_embedding_model()

    # Assert that the returned object is not None.
    assert embeddings is not None

    # Assert that the returned object is an instance of HuggingFaceEmbeddings.
    assert isinstance(embeddings, HuggingFaceEmbeddings)

def test_create_faiss_index():
    # Create some dummy documents for testing.
    documents = [
        Document(page_content="This is the first document.", metadata={"source": "doc1.txt"}),
        Document(page_content="This is the second document.", metadata={"source": "doc2.txt"}),
        Document(page_content="And this is the third one.", metadata={"source": "doc3.txt"}),
    ]

    # Initialize the embedding model.
    embeddings = initialize_embedding_model()

    # Create the FAISS index.
    faiss_index = create_faiss_index(documents, embeddings)

    # Assert that the returned object is a FAISS instance.
    assert isinstance(faiss_index, FAISS)

    # Assert that the index contains the correct number of vectors.
    assert faiss_index.index.ntotal == len(documents)

def test_save_load_faiss_index():
    # Create a temporary directory to save the FAISS index.
    with tempfile.TemporaryDirectory() as tmpdir:
        index_path = os.path.join(tmpdir, "test_faiss_index")

        # Create some dummy documents for the index.
        documents = [
            Document(page_content="Apple is a fruit.", metadata={"id": 1}),
            Document(page_content="Banana is yellow.", metadata={"id": 2}),
            Document(page_content="Cherry is red.", metadata={"id": 3}),
        ]

        # Initialize the embedding model.
        embeddings = initialize_embedding_model()

        # Create the initial FAISS index.
        original_faiss_index = create_faiss_index(documents, embeddings)

        # Save the FAISS index.
        save_faiss_index(original_faiss_index, index_path)

        # Load the FAISS index.
        loaded_faiss_index = load_faiss_index(index_path, embeddings)

        # Assert that the loaded index is a FAISS instance.
        assert isinstance(loaded_faiss_index, FAISS)

        # Assert that the loaded index has the same number of vectors as the original.
        assert loaded_faiss_index.index.ntotal == original_faiss_index.index.ntotal

        # Perform a similarity search on both indexes to ensure they are functionally equivalent.
        query = "What fruit is red?"
        # Get top 1 similar document from original index
        docs_original = original_faiss_index.similarity_search(query, k=1)
        # Get top 1 similar document from loaded index
        docs_loaded = loaded_faiss_index.similarity_search(query, k=1)

        # Assert that the content of the top retrieved document is the same.
        assert docs_original[0].page_content == docs_loaded[0].page_content
        # Assert that the metadata of the top retrieved document is the same.
        assert docs_original[0].metadata == docs_loaded[0].metadata

def test_load_faiss_index_nonexistent_path():
    # Define a non-existent path for the FAISS index.
    non_existent_path = "./non_existent_faiss_index"
    # Initialize a dummy embedding model.
    embeddings = initialize_embedding_model()

    try:
        # Expect a FileNotFoundError when trying to load from a non-existent path.
        load_faiss_index(non_existent_path, embeddings)
        # If no error is raised, fail the test.
        assert False, "FileNotFoundError was not raised for a non-existent FAISS index path."
    except FileNotFoundError as e:
        # Assert that the error message is as expected.
        assert f"FAISS index not found at: {non_existent_path}" in str(e)
