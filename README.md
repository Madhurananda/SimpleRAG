# 🧠 Enterprise RAG Platform

![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-1C3C3C?logo=langchain&logoColor=white)
![Groq](https://img.shields.io/badge/Groq-F55036?logo=groq&logoColor=white)
![FAISS](https://img.shields.io/badge/FAISS-00599C)
![ChromaDB](https://img.shields.io/badge/ChromaDB-5A67D8)
![LangSmith](https://img.shields.io/badge/LangSmith-00A67E)
![LangFuse](https://img.shields.io/badge/LangFuse-FFCC00)
![Jina_AI](https://img.shields.io/badge/Jina_AI-EF4E23)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-2496ED?logo=docker&logoColor=white)
![Git](https://img.shields.io/badge/Git-F05032?logo=git&logoColor=white)
![GitHub](https://img.shields.io/badge/GitHub-181717?logo=github&logoColor=white)

A modular, production-style **Retrieval-Augmented Generation (RAG)** platform built with **LangChain**, **Groq**, **FAISS/ChromaDB**, **Streamlit**, and **Docker**.

The project demonstrates how to design scalable AI assistants using configurable retrieval pipelines, semantic search, conversational memory, vector databases, and modern LLM engineering practices.

## 🚀 Current Applications

### 📄 HR Policy Assistant
An enterprise knowledge assistant that answers questions about HR policies using Retrieval-Augmented Generation.

Features:
- Semantic document search
- Conversational memory
- Source-grounded responses
- Persistent vector database
- Streamlit chat interface

---

### 🎤 Interspeech Research Assistant

An academic research assistant that searches and summarizes Interspeech conference papers.

Features:
- Web scraping pipeline
- Parallel data collection
- Persistent JSON caching
- Year filtering
- Dementia paper filtering
- Citation-aware retrieval
- Automatic vector index rebuilding

---

## ✨ Key Features

- Modular RAG architecture
- Independent AI assistants
- LangChain agent orchestration
- FAISS and ChromaDB support
- Groq LLM integration
- Jina AI and local embedding models
- Conversational memory
- Configurable retrieval pipelines
- LangSmith tracing
- LangFuse observability
- Docker deployment
- Environment-based configuration
- Persistent vector indexes
- Cached data pipelines

---

# 🏗️ Architecture

```
                 User
                   │
          Streamlit Web App
                   │
           LangChain Agent
                   │
        Retrieval + Tool Calling
                   │
        FAISS / ChromaDB Vector Store
                   │
     Groq LLM + Embedding Model
                   │
             Final Response
```

---

# 🛠️ Technology Stack

| Category | Technologies |
|-----------|--------------|
| **Language** | Python 3.11 |
| **Web Framework** | Streamlit |
| **LLM Framework** | LangChain |
| **Large Language Model** | Groq (Llama 3.1) |
| **Embeddings** | Jina AI, Sentence Transformers |
| **Vector Databases** | FAISS, ChromaDB |
| **Observability** | LangSmith, LangFuse |
| **Web Scraping** | BeautifulSoup, Requests |
| **Data Processing** | Pandas, NumPy |
| **Configuration** | Python dotenv |
| **Containerization** | Docker, Docker Compose |
| **Version Control** | Git, GitHub |
| **Cloud Deployment** | Azure (planned), AWS Compatible, Google Cloud Compatible |

---

# 🤖 AI Stack

### LLM
- Groq
- Llama 3.1

### AI Framework
- LangChain

### Retrieval-Augmented Generation
- Document Chunking
- Semantic Search
- Conversational Memory
- Source Attribution

### Embeddings
- Jina AI
- Sentence Transformers

### Vector Databases
- FAISS
- ChromaDB

### Observability
- LangSmith
- LangFuse

---

# 📂 Project Structure

```text
enterprise-rag-platform/

├── app.py                         # HR Assistant
├── app_interspeech.py             # Interspeech Assistant
├── app_unified.py                 # Unified application
│
├── hr_assistant/
│   ├── pipeline.py
│   ├── document_loader.py
│   └── tools.py
│
├── interspeech/
│   ├── pipeline.py
│   ├── scraper.py
│   ├── loader.py
│   └── tools.py
│
├── src/
│   ├── config.py
│   ├── llm.py
│   ├── embeddings.py
│   ├── vector_store.py
│   ├── splitter.py
│   └── agent.py
│
├── data/
│
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

---

# ⚡ Quick Start

## Clone Repository

```bash
git clone https://github.com/yourusername/enterprise-rag-platform.git

cd enterprise-rag-platform
```

---

## Create Environment

```bash
python -m venv venv

source venv/bin/activate

# Windows

venv\Scripts\activate
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Configure Environment Variables

Create a `.env` file.

Example:

```env
GROQ_API_KEY=your_key
JINA_API_KEY=your_key

LANGCHAIN_API_KEY=optional
LANGFUSE_PUBLIC_KEY=optional
LANGFUSE_SECRET_KEY=optional
```

---

## Run

HR Assistant

```bash
streamlit run app.py
```

Interspeech Assistant

```bash
streamlit run app_interspeech.py
```

Unified Application

```bash
streamlit run app_unified.py
```

---

# 🐳 Docker

Build and start the application

```bash
docker compose up --build
```

---

# 🔍 Engineering Highlights

This project demonstrates experience with:

- Retrieval-Augmented Generation (RAG)
- LangChain Agents
- Tool Calling
- Prompt Engineering
- Semantic Search
- Vector Databases
- Conversational Memory
- AI Observability
- LangSmith Tracing
- LangFuse Monitoring
- Persistent Vector Storage
- Web Scraping Pipelines
- Docker Containerization
- Config-Driven Architecture
- Production-Style Project Structure

---

# 📈 Resume Skills Demonstrated

### AI & Machine Learning
- Retrieval-Augmented Generation (RAG)
- Large Language Models (LLMs)
- LangChain
- Prompt Engineering
- Semantic Search
- Information Retrieval
- Embedding Models
- AI Application Development

### AI Platforms
- Groq
- LangSmith
- LangFuse
- Jina AI
- Hugging Face Sentence Transformers

### Backend Development
- Python
- REST APIs
- Streamlit
- Configuration Management

### Vector Databases
- FAISS
- ChromaDB

### Data Engineering
- BeautifulSoup
- Requests
- Pandas
- NumPy
- JSON Processing
- Parallel Data Collection

### DevOps
- Docker
- Docker Compose
- Environment Variables (.env)
- Git
- GitHub

---

# 🔮 Future Improvements

- LangGraph workflows
- Hybrid Search (BM25 + Dense Retrieval)
- Evaluation framework
- User authentication
- Azure deployment
- Additional domain-specific assistants

---

# 📄 License

This project is licensed under the MIT License.

---

## ⭐ Support

If you found this project useful, consider giving it a ⭐ on GitHub.
