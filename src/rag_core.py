from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from typing import List
import os

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
    # Use a pre-trained sentence-transformer model. 'all-MiniLM-L6-v2' is a good default
    # for its balance of performance and size, suitable for local execution.
    # The model is downloaded and loaded locally.
    model_name = "all-MiniLM-L6-v2"
    model_kwargs = {'device': 'cpu'} # Specify to run on CPU, as GPU might not always be available.
    encode_kwargs = {'normalize_embeddings': False} # Normalizing embeddings can be beneficial for cosine similarity.

    # Initialize the HuggingFaceEmbeddings object.
    # This object will be used to generate embeddings for text.
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    return embeddings

# Placeholder for the embedding model.
# This variable will hold the initialized embedding model.
embedding_model = initialize_embedding_model() # Initialize the embedding model upon module load.

# Placeholder for the FAISS vector store.
# This section will be replaced with actual FAISS index creation and management
# in a later step. FAISS (Facebook AI Similarity Search) is an efficient library
# for similarity search and clustering of dense vectors, enabling fast retrieval
# of relevant documents based on query embeddings.
faiss_index = None  # To be initialized later with a FAISS index