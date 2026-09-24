"""Step 1: read the raw HR document."""

from langchain_community.document_loaders import TextLoader


def load_document(file_path: str):
    """Load a .txt file and return a list of LangChain Document objects."""
    return TextLoader(file_path, encoding="utf-8").load()
