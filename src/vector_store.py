# """Step 4: store chunk embeddings in FAISS so we can search them later."""


# import os
# from langchain_community.vectorstores import FAISS

# from src import config 
# from src.embeddings import get_embeddings_model


# # build_vector_store 

# def build_vector_store(chunks):
#     """Embed every chunk and build 
#     a searchable FAISS index in memory."""
#     embeddings_model = get_embeddings_model()
#     return FAISS.from_documents(chunks, embeddings_model)


# ## save vector store 

# def save_vector_store(vector_store, path: str = config.VECTOR_STORE_PATH) -> None:
#     """Save the FAISS index to disk 
#     so we don't have to rebuild it every time."""
#     vector_store.save_local(path)


# def load_vector_store(path: str = config.VECTOR_STORE_PATH):
#     """Load a previously saved FAISS index from disk."""
#     embeddings_model = get_embeddings_model()
#     # allow_dangerous_deserialization is safe here because we only ever load
#     # an index that this same app created and saved.
#     return FAISS.load_local(path, embeddings_model, allow_dangerous_deserialization=True)


# def vector_store_exists(path: str = config.VECTOR_STORE_PATH) -> bool:
#     """Check if a saved FAISS index already exists on disk."""
#     return os.path.exists(os.path.join(path, "index.faiss"))


# def get_retriever(vector_store, k: int = 3):
#     """Turn a vector store into a retriever 
#     that returns the top-k matching chunks."""
#     return vector_store.as_retriever(search_kwargs={"k": k})



"""Step 4: store chunk embeddings in FAISS or ChromaDB (toggle via config)."""

import os
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores import Chroma

from src import config
from src.embeddings import get_embeddings_model


def build_vector_store(chunks):
    """Build a vector store (FAISS or ChromaDB) based on config."""
    embeddings_model = get_embeddings_model()
    
    if config.VECTOR_STORE_TYPE == "faiss":
        print("🔄 Building FAISS index...")
        return FAISS.from_documents(chunks, embeddings_model)
    
    elif config.VECTOR_STORE_TYPE == "chromadb":
        print("🔄 Building ChromaDB index...")
        return Chroma.from_documents(
            documents=chunks,
            embedding=embeddings_model,
            persist_directory=config.VECTOR_STORE_PATH  # This is the ChromaDB directory
        )
    
    else:
        raise ValueError(f"Unknown VECTOR_STORE_TYPE: {config.VECTOR_STORE_TYPE}")


def save_vector_store(vector_store, path: str = config.VECTOR_STORE_PATH) -> None:
    """Save the vector store to disk."""
    if config.VECTOR_STORE_TYPE == "faiss":
        print(f"💾 Saving FAISS index to {path}...")
        vector_store.save_local(path)
    
    elif config.VECTOR_STORE_TYPE == "chromadb":
        print(f"💾 Persisting ChromaDB to {path}...")
        # ChromaDB automatically persists when you call persist()
        # But if you want to be explicit:
        vector_store.persist()
        print(f"✅ ChromaDB persisted to: {vector_store._persist_directory}")
    
    else:
        raise ValueError(f"Unknown VECTOR_STORE_TYPE: {config.VECTOR_STORE_TYPE}")


def load_vector_store(path: str = config.VECTOR_STORE_PATH):
    """Load a previously saved vector store from disk."""
    embeddings_model = get_embeddings_model()
    
    if config.VECTOR_STORE_TYPE == "faiss":
        print(f"📂 Loading FAISS from {path}...")
        return FAISS.load_local(
            path,
            embeddings_model,
            allow_dangerous_deserialization=True
        )
    
    elif config.VECTOR_STORE_TYPE == "chromadb":
        print(f"📂 Loading ChromaDB from {path}...")
        return Chroma(
            persist_directory=path,
            embedding_function=embeddings_model
        )
    
    else:
        raise ValueError(f"Unknown VECTOR_STORE_TYPE: {config.VECTOR_STORE_TYPE}")


def vector_store_exists(path: str = config.VECTOR_STORE_PATH) -> bool:
    """Check if a saved vector store exists."""
    if config.VECTOR_STORE_TYPE == "faiss":
        return os.path.exists(os.path.join(path, "index.faiss"))
    
    elif config.VECTOR_STORE_TYPE == "chromadb":
        # ChromaDB stores its data in a SQLite file
        return os.path.exists(os.path.join(path, "chroma.sqlite3"))
    
    else:
        return False


def get_retriever(vector_store, k: int = 3):
    """Turn a vector store into a retriever."""
    return vector_store.as_retriever(search_kwargs={"k": k})