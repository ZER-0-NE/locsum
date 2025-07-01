from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS # Import FAISS for vector store
from typing import List
import os
import torch # Import torch to check for GPU availability

# Define the global constant for the embedding model name.
# This allows for easy modification of the model used across the application.
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# This function is responsible for loading documents from a specified directory.
# It is designed to be modular, allowing for future integration with an Obsidian MCP server.
def load_documents_from_directory(directory_path: str) -> List[Document]:
    """
    Loads markdown documents from a specified directory.

    Args:
        directory_path (str): The absolute path to the directory containing the markdown files.

    Returns:
        List[Document]: A list of LangChain Document objects, each representing a loaded markdown file.
    """
    # Ensure the provided directory path exists before attempting to load documents.
    if not os.path.exists(directory_path):
        # If the directory does not exist, raise an error to inform the caller.
        raise FileNotFoundError(f"Directory not found: {directory_path}")

    # Initialize a DirectoryLoader to load markdown files.
    # The 'glob' pattern '**/*.md' ensures that all markdown files
    # within the specified directory and its subdirectories are included.
    # The 'loader_cls' is set to 'TextLoader' as markdown files are essentially text files.
    loader = DirectoryLoader(directory_path, glob="**/*.md", loader_cls=TextLoader)

    # Load the documents using the configured loader.
    # This process reads the content of each markdown file and converts it
    # into a LangChain Document object.
    documents = loader.load()

    # Return the list of loaded Document objects.
    return documents

# This function is responsible for splitting documents into smaller, manageable chunks.
# This is crucial for RAG systems as LLMs have context window limitations,
# and smaller chunks allow for more precise retrieval.
def chunk_documents(documents: List[Document], chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Document]:
    """
    Splits a list of documents into smaller chunks.

    Args:
        documents (List[Document]): A list of LangChain Document objects to be chunked.
        chunk_size (int): The maximum size of each chunk.
        chunk_overlap (int): The number of characters to overlap between consecutive chunks.

    Returns:
        List[Document]: A list of chunked LangChain Document objects.
    """
    # Initialize the RecursiveCharacterTextSplitter.
    # This splitter attempts to split text in a way that keeps sentences and paragraphs together,
    # using a list of characters to split on.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,  # Define the maximum size for each text chunk.
        chunk_overlap=chunk_overlap,  # Define the overlap between chunks to maintain context.
        length_function=len,  # Use the standard Python len() function to measure chunk length.
        add_start_index=True, # Add a metadata field for the starting index of each chunk.
    )

    # Split the documents into chunks.
    # The 'split_documents' method processes each document in the list
    # and applies the splitting logic.
    chunked_documents = text_splitter.split_documents(documents)

    # Return the list of chunked Document objects.
    return chunked_documents

# This function initializes the embedding model.
# The embedding model converts text into numerical vectors (embeddings),
# which are essential for calculating similarity between queries and documents.
def initialize_embedding_model():
    """
    Initializes the HuggingFace embedding model.

    Returns:
        HuggingFaceEmbeddings: An initialized HuggingFaceEmbeddings object.
    """
    # Determine the appropriate device for the embedding model.
    # Prioritize CUDA (NVIDIA GPUs), then MPS (Apple Silicon GPUs), and finally CPU.
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    # Define model keyword arguments, including the determined device.
    model_kwargs = {'device': device}
    # Define encoding keyword arguments. Normalizing embeddings can be beneficial for cosine similarity.
    encode_kwargs = {'normalize_embeddings': False}

    # Initialize the HuggingFaceEmbeddings object using the global model name and determined device.
    # This object will be used to generate embeddings for text.
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    return embeddings

# This function creates a FAISS vector store from a list of documents and an embedding model.
# FAISS is used for efficient similarity search of vector embeddings.
def create_faiss_index(documents: List[Document], embeddings: HuggingFaceEmbeddings):
    """
    Creates a FAISS vector store from a list of documents and an embedding model.

    Args:
        documents (List[Document]): A list of LangChain Document objects to be indexed.
        embeddings (HuggingFaceEmbeddings): The embedding model to use for generating document embeddings.

    Returns:
        FAISS: A FAISS vector store object.
    """
    # Create the FAISS index from the documents and embeddings.
    # This process embeds each document and adds its vector to the FAISS index.
    vector_store = FAISS.from_documents(documents, embeddings)
    return vector_store

# Initialize the embedding model upon module load.
embedding_model = initialize_embedding_model()

# Placeholder for the FAISS vector store.
# This variable will hold the initialized FAISS index.
# It is initialized to None, and will be created dynamically when documents are available.
faiss_index = None # Initialize to None, will be created when documents are processed.
