"""Recursive text splitting into overlapping chunks."""

from langchain_text_splitters import RecursiveCharacterTextSplitter
from core import config


def split_into_chunks(documents, chunk_size: int = config.CHUNK_SIZE, chunk_overlap: int = config.CHUNK_OVERLAP):
    """Split LangChain documents into overlapping chunks ready for embedding."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_documents(documents)

