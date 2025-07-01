import os
import shutil
from src.rag_core import load_documents_from_directory

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