from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from typing import List, Dict, Any
import os
import torch
import hashlib
import json

# Define the global constant for the embedding model name.
EMBEDDING_MODEL_NAME = "all-mpnet-base-v2"

# Define the global constant for the Ollama LLM model name.
OLLAMA_LLM_MODEL_NAME = "llama3"

def create_index_manifest(documents: List[Document], chunk_size: int, chunk_overlap: int) -> Dict[str, Any]:
    """
    Creates a manifest dictionary containing a signature of the data and configuration.

    Args:
        documents (List[Document]): The list of documents to be indexed.
        chunk_size (int): The chunk size used for splitting documents.
        chunk_overlap (int): The chunk overlap used for splitting documents.

    Returns:
        Dict[str, Any]: A dictionary representing the manifest.
    """
    # Hash the content of the rag_core.py file itself to detect code changes.
    with open(__file__, 'rb') as f:
        rag_core_hash = hashlib.md5(f.read()).hexdigest()

    # Create a hash of all document content.
    # Sorting by page content ensures a consistent hash regardless of document order.
    doc_content_str = "".join(sorted([doc.page_content for doc in documents]))
    data_hash = hashlib.md5(doc_content_str.encode()).hexdigest()

    return {
        "data_hash": data_hash,
        "rag_core_hash": rag_core_hash,
        "embedding_model": EMBEDDING_MODEL_NAME,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap
    }

def generate_directory_summary(directory_path: str) -> Document:
    """
    Walks a directory to create a text summary of its structure and file contents.

    Args:
        directory_path (str): The absolute path to the directory to summarize.

    Returns:
        Document: A LangChain Document object containing the summary.
    """
    summary_lines = ["# Vault Structure Summary", "This document provides a summary of the files and folders in the vault."]
    dir_file_counts = {}

    for root, dirs, files in os.walk(directory_path):
        # Don't include the root directory itself in the summary
        if root == directory_path:
            continue
        
        # Get relative path from the vault root
        relative_path = os.path.relpath(root, directory_path)
        dir_file_counts[relative_path] = {
            'file_count': 0,
            'files': []
        }

        for file in files:
            if file.endswith('.md'):
                dir_file_counts[relative_path]['file_count'] += 1
                dir_file_counts[relative_path]['files'].append(file)

    for path, data in sorted(dir_file_counts.items()):
        if data['file_count'] > 0:
            summary_lines.append(f"\n## Directory: {path}")
            summary_lines.append(f"- Contains {data['file_count']} markdown file(s).")
            summary_lines.append("- Files:")
            for file_name in sorted(data['files']):
                summary_lines.append(f"  - {file_name}")

    summary_content = "\n".join(summary_lines)
    return Document(page_content=summary_content, metadata={"source": "Generated Directory Summary"})

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
    Splits a list of documents into smaller chunks and prepends metadata to each chunk.

    Args:
        documents (List[Document]): A list of LangChain Document objects to be chunked.
        chunk_size (int): The maximum size of each chunk.
        chunk_overlap (int): The number of characters to overlap between consecutive chunks.

    Returns:
        List[Document]: A list of chunked LangChain Document objects with metadata prepended.
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

    # Prepend metadata to the content of each chunk.
    # This makes the file path and other metadata searchable within the RAG system.
    for chunk in chunked_documents:
        source_path = chunk.metadata.get('source', 'Unknown Source')
        chunk.page_content = f"Source: {source_path}\n\n{chunk.page_content}"

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

# This function saves a FAISS index to disk.
# Persisting the index avoids re-embedding all documents on every application restart.
def save_faiss_index(faiss_index: FAISS, path: str):
    """
    Saves a FAISS index to the specified path.

    Args:
        faiss_index (FAISS): The FAISS index to save.
        path (str): The file path where the index will be saved.
    """
    # Ensure the directory for the save path exists.
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # Save the FAISS index to disk.
    faiss_index.save_local(path)

# This function loads a FAISS index from disk.
# It requires the embedding model to be the same as the one used for creation.
def load_faiss_index(path: str, embeddings: HuggingFaceEmbeddings) -> FAISS:
    """
    Loads a FAISS index from the specified path.

    Args:
        path (str): The file path from which to load the index.
        embeddings (HuggingFaceEmbeddings): The embedding model used to create the index.

    Returns:
        FAISS: The loaded FAISS index object.
    """
    # Ensure the index file exists before attempting to load.
    if not os.path.exists(path):
        raise FileNotFoundError(f"FAISS index not found at: {path}")
    # Load the FAISS index from disk.
    faiss_index = FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)
    return faiss_index

# This function creates a retriever from a FAISS vector store.
# The retriever is responsible for fetching relevant documents based on a query.
def get_retriever(faiss_index: FAISS):
    """
    Creates a retriever from a FAISS vector store.

    Args:
        faiss_index (FAISS): The FAISS vector store.

    Returns:
        VectorStoreRetriever: A LangChain VectorStoreRetriever object.
    """
    # Convert the FAISS index into a retriever.
    # The 'as_retriever()' method allows the vector store to be used for document retrieval.
    retriever = faiss_index.as_retriever()
    return retriever

# This function initializes the Ollama LLM.
# Ollama allows running large language models locally.
def get_llm():
    """
    Initializes the Ollama Large Language Model.

    Returns:
        Ollama: An initialized Ollama LLM object.
    """
    # Initialize the Ollama LLM with the specified model name.
    # Ensure that the Ollama server is running and the model is downloaded.
    llm = Ollama(model=OLLAMA_LLM_MODEL_NAME)
    return llm

# This function constructs the RAG (Retrieval Augmented Generation) chain.
# The chain combines document retrieval with LLM generation to answer queries.
def get_rag_chain(retriever, llm):
    """
    Constructs the RAG (Retrieval Augmented Generation) chain.

    Args:
        retriever: The document retriever (e.g., from FAISS).
        llm: The Large Language Model (e.g., Ollama).

    Returns:
        Runnable: A LangChain Runnable object representing the RAG chain.
    """
    # Define the prompt template for the LLM.
    # The prompt guides the LLM to use the provided context for answering the question.
    template = """Answer the question based only on the following context:
{context}

Question: {question}
"""
    prompt = ChatPromptTemplate.from_template(template)

    # Construct the RAG chain using LangChain's LCEL (LangChain Expression Language).
    # The chain first retrieves relevant documents, then formats them into the prompt,
    # passes the prompt to the LLM, and finally parses the LLM's output.
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()} # Retrieve context and pass through question
        | prompt # Apply the prompt template
        | llm # Invoke the LLM
        | StrOutputParser() # Parse the LLM's output to a string
    )
    return rag_chain

# Initialize the embedding model upon module load.
embedding_model = initialize_embedding_model()

# Placeholder for the FAISS vector store.
# This variable will hold the initialized FAISS index.
# It is initialized to None, and will be created dynamically when documents are available.
faiss_index = None # Initialize to None, will be created when documents are processed.
