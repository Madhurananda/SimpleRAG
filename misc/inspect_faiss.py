"""Inspect the text lengths stored in the FAISS index."""

from src import config
from src.vector_store import load_vector_store
from src.embeddings import get_embeddings_model

def inspect_faiss():
    print("=" * 60)
    print("🔍 Inspecting FAISS Index")
    print("=" * 60)

    # Load the vector store
    vector_store_path = config.INTERSPEECH_VECTOR_STORE_PATH
    print(f"\n📁 Loading FAISS from: {vector_store_path}")

    try:
        embeddings = get_embeddings_model()
        vector_store = load_vector_store(path=vector_store_path)
    except Exception as e:
        print(f"❌ Error loading FAISS: {e}")
        return

    # Get the underlying store
    store = vector_store.docstore._dict  # Access the stored documents
    print(f"\n📊 Total chunks in FAISS: {len(store)}")

    # Show sample chunks
    print("\n📄 Sample chunk contents:")
    print("-" * 40)
    
    lengths = []
    for i, (doc_id, doc) in enumerate(list(store.items())[:5]):
        text = doc.page_content
        lengths.append(len(text))
        print(f"Chunk {i+1}:")
        print(f"  Length: {len(text)} characters")
        print(f"  Preview: {text[:100]}...")
        print("-" * 20)

    # Statistics
    if lengths:
        import statistics
        print("\n📊 Chunk statistics:")
        print(f"  Min length: {min(lengths)} chars")
        print(f"  Max length: {max(lengths)} chars")
        print(f"  Avg length: {sum(lengths)/len(lengths):.0f} chars")

    print("\n✅ Done.")

if __name__ == "__main__":
    inspect_faiss()
