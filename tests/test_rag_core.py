import os
import shutil
from src.rag_core import load_documents_from_directory, chunk_documents, initialize_embedding_model
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings

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
