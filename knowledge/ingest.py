"""
Knowledge ingestion pipeline for banking policy documents.
Loads markdown policy docs, chunks them, generates embeddings,
and stores in ChromaDB for RAG retrieval.
"""

import os
from pathlib import Path
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


POLICIES_DIR = Path(__file__).parent / "policies"
CHROMA_DIR = Path(__file__).parent.parent / ".chroma"


def load_policies() -> list:
    """Load all markdown policy documents from the policies directory."""
    loader = DirectoryLoader(
        str(POLICIES_DIR),
        glob="**/*.md",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
    )
    documents = loader.load()
    print(f"Loaded {len(documents)} policy documents")
    return documents


def chunk_documents(documents: list, chunk_size: int = 500, chunk_overlap: int = 100) -> list:
    """
    Split documents into chunks optimized for retrieval.

    Uses recursive character splitting with markdown-aware separators
    to preserve document structure (headers, sections, lists).
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n## ", "\n### ", "\n- ", "\n\n", "\n", " "],
        keep_separator=True,
    )
    chunks = splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks from {len(documents)} documents")
    return chunks


def build_vectorstore(chunks: list, persist_directory: str = None) -> Chroma:
    """
    Generate embeddings and store in ChromaDB.

    Uses HuggingFace all-MiniLM-L6-v2 for local embedding generation.
    Persists to disk for reuse across sessions.
    """
    persist_dir = persist_directory or str(CHROMA_DIR)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_dir,
        collection_name="banking_policies",
    )
    print(f"Vectorstore built with {len(chunks)} chunks at {persist_dir}")
    return vectorstore


def load_vectorstore(persist_directory: str = None) -> Chroma:
    """Load an existing vectorstore from disk."""
    persist_dir = persist_directory or str(CHROMA_DIR)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    return Chroma(
        persist_directory=persist_dir,
        embedding_function=embeddings,
        collection_name="banking_policies",
    )


def get_retriever(k: int = 4):
    """
    Return a retriever from the vectorstore.
    Builds the store if it doesn't exist yet.
    """
    if not CHROMA_DIR.exists():
        print("Vectorstore not found — building from policy documents...")
        docs = load_policies()
        chunks = chunk_documents(docs)
        vectorstore = build_vectorstore(chunks)
    else:
        vectorstore = load_vectorstore()

    return vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k},
    )


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    docs = load_policies()
    chunks = chunk_documents(docs)
    vectorstore = build_vectorstore(chunks)
    print("Knowledge base built successfully.")
