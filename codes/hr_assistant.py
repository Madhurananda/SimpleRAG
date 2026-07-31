# import os
# from dotenv import load_dotenv

# # LangChain imports
# from langchain_community.document_loaders import TextLoader
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_community.embeddings import JinaEmbeddings
# from langchain_community.vectorstores import FAISS
# from langchain_groq import ChatGroq
# from langchain.agents import create_agent

# # Load environment variables
# load_dotenv()

# # Initialize API keys
# groq_key = os.getenv("GROQ_API_KEY")
# jina_key = os.getenv("JINA_API_KEY")

# # ------------------------------------------------------------
# # STEP 1: LOAD DATA
# # ------------------------------------------------------------
# def load_data(file_path="data/hr_policy.txt"):
#     loader = TextLoader(file_path, encoding="utf-8")
#     documents = loader.load()
#     print(f"✅ Loaded {len(documents)} document with {len(documents[0].page_content)} characters")
#     return documents

# # ------------------------------------------------------------
# # STEP 2: SPLIT INTO CHUNKS
# # ------------------------------------------------------------
# def split_documents(documents):
#     text_splitter = RecursiveCharacterTextSplitter(
#         chunk_size=500,
#         chunk_overlap=50
#     )
#     chunks = text_splitter.split_documents(documents)
#     print(f"✅ Split into {len(chunks)} chunks")
#     return chunks

# # ------------------------------------------------------------
# # STEP 3: EMBED AND STORE IN VECTOR DB
# # ------------------------------------------------------------
# def create_vector_store(chunks):
#     embeddings = JinaEmbeddings(model_name="jina-embeddings-v2-base-en")
#     vector_store = FAISS.from_documents(chunks, embeddings)
#     print(f"✅ Vector store created with {vector_store.index.ntotal} vectors")
#     return vector_store

# # ------------------------------------------------------------
# # STEP 4: CREATE RETRIEVER AND TOOL
# # ------------------------------------------------------------
# def create_retriever(vector_store, k=3):
#     retriever = vector_store.as_retriever(search_kwargs={"k": k})
    
#     def search_hr_policy(question: str) -> str:
#         """Search the HR policy document for information."""
#         matching_chunks = retriever.invoke(question)
#         return "\n\n".join(chunk.page_content for chunk in matching_chunks)
    
#     return search_hr_policy

# # ------------------------------------------------------------
# # STEP 5: CREATE THE AI AGENT
# # ------------------------------------------------------------
# def create_hr_agent(tool):
#     llm = ChatGroq(
#         model="openai/gpt-oss-120b",
#         temperature=0.5
#     )
    
#     agent = create_agent(
#         model=llm,
#         tools=[tool],
#         system_prompt="""
#         You are a friendly HR assistant working for Acme Corp. 
#         Always use the search_hr_policy tool to look up facts before answering. 
#         If the answer isn't in the search results, say you don't know instead of guessing.
#         """
#     )
#     print("✅ HR Assistant agent ready!")
#     return agent

# # ------------------------------------------------------------
# # STEP 6: MAIN EXECUTION (Runs when you call python hr_assistant.py)
# # ------------------------------------------------------------
# if __name__ == "__main__":
#     print("🚀 Starting HR Policy RAG Assistant...")
#     print("-" * 60)
    
#     # Run the pipeline
#     docs = load_data()
#     chunks = split_documents(docs)
#     vector_store = create_vector_store(chunks)
#     search_tool = create_retriever(vector_store)
#     agent = create_hr_agent(search_tool)
    
#     print("-" * 60)
#     print("💬 Ready to answer questions! Type 'exit' to quit.\n")
    
#     # Interactive Q&A loop
#     while True:
#         question = input("❓ Ask a question: ")
#         if question.lower() in ["exit", "quit", "q"]:
#             print("👋 Goodbye!")
#             break
        
#         response = agent.invoke({"messages": [{"role": "user", "content": question}]})
#         answer = response["messages"][-1].content
#         print(f"🤖 Answer: {answer}\n")





import os
from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import JinaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.agents import create_agent

load_dotenv()

groq_key = os.getenv("GROQ_API_KEY")
jina_key = os.getenv("JINA_API_KEY")

# ------------------------------------------------------------
# STEP 1: LOAD DATA
# ------------------------------------------------------------
def load_data(file_path="data/hr_policy.txt"):
    loader = TextLoader(file_path, encoding="utf-8")
    documents = loader.load()
    print(f"✅ Loaded {len(documents)} document with {len(documents[0].page_content)} characters")
    return documents

# ------------------------------------------------------------
# STEP 2: SPLIT INTO CHUNKS
# ------------------------------------------------------------
def split_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = text_splitter.split_documents(documents)
    print(f"✅ Split into {len(chunks)} chunks")
    return chunks

# ------------------------------------------------------------
# STEP 3: EMBED AND STORE IN VECTOR DB
# ------------------------------------------------------------
def create_vector_store(chunks):
    embeddings = JinaEmbeddings(model_name="jina-embeddings-v2-base-en")
    vector_store = FAISS.from_documents(chunks, embeddings)
    print(f"✅ Vector store created with {vector_store.index.ntotal} vectors")
    return vector_store

# ------------------------------------------------------------
# STEP 4: CREATE RETRIEVER AND TOOL
# ------------------------------------------------------------
def create_retriever(vector_store, k=3):
    retriever = vector_store.as_retriever(search_kwargs={"k": k})
    
    def search_hr_policy(question: str) -> str:
        """Search the HR policy document for information."""
        matching_chunks = retriever.invoke(question)
        return "\n\n".join(chunk.page_content for chunk in matching_chunks)
    
    # <--- NEW: return both the retriever object and the tool function
    return retriever, search_hr_policy

# ------------------------------------------------------------
# STEP 5: CREATE THE AI AGENT
# ------------------------------------------------------------
def create_hr_agent(tool):
    llm = ChatGroq(
        model="openai/gpt-oss-120b",
        temperature=0.5
    )
    
    agent = create_agent(
        model=llm,
        tools=[tool],
        system_prompt="""
        You are a friendly HR assistant working for Acme Corp. 
        Always use the search_hr_policy tool to look up facts before answering. 
        If the answer isn't in the search results, say you don't know instead of guessing.
        """
    )
    print("✅ HR Assistant agent ready!")
    return agent

# ------------------------------------------------------------
# STEP 6: MAIN EXECUTION
# ------------------------------------------------------------
if __name__ == "__main__":
    print("🚀 Starting HR Policy RAG Assistant...")
    print("-" * 60)
    
    # Run the pipeline
    docs = load_data()
    chunks = split_documents(docs)
    vector_store = create_vector_store(chunks)
    
    # <--- NEW: unpack retriever and tool
    retriever, search_tool = create_retriever(vector_store)
    
    agent = create_hr_agent(search_tool)
    
    print("-" * 60)
    print("💬 Ready to answer questions! Type 'exit' to quit.\n")
    
    # Interactive Q&A loop
    while True:
        question = input("❓ Ask a question: ")
        if question.lower() in ["exit", "quit", "q"]:
            print("👋 Goodbye!")
            break
        
        # <--- NEW: Show top-3 raw chunks before the LLM answers
        raw_chunks = retriever.invoke(question)
        print("\n📄 Top-3 matching chunks (raw retrieval):")
        for i, chunk in enumerate(raw_chunks, 1):
            print(f"Chunk {i}:")
            # Print first 300 chars to keep output readable, but you can print full
            print(chunk.page_content[:300] + ("..." if len(chunk.page_content) > 300 else ""))
            print(f"Source: {chunk.metadata.get('source', 'N/A')}")
            print("-" * 40)
        print()  # blank line
        
        # Now the agent uses the same chunks via the tool
        response = agent.invoke({"messages": [{"role": "user", "content": question}]})
        answer = response["messages"][-1].content
        print(f"🤖 Answer (after LLM): {answer}\n")