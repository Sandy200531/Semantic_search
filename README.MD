# Semantic Document Search

A semantic search project that uses **text embeddings** and **vector databases** to find documents based on their meaning rather than exact keyword matching.

## Features

- Generate text embeddings using **Sentence Transformers**
- Store and search vectors using **FAISS**
- Store embeddings and documents using **ChromaDB**
- Perform semantic similarity search
- Compare relevant documents based on vector similarity

## Technologies Used

- Python
- Sentence Transformers
- NumPy
- FAISS
- ChromaDB

## How It Works

```text
Documents
    ↓
Sentence Transformer
    ↓
Text Embeddings
    ↓
Vector Storage
    ├── FAISS
    └── ChromaDB
    ↓
User Query
    ↓
Query Embedding
    ↓
Similarity Search
    ↓
Top 3 Relevant Documents
```

## Embedding Model

The project uses:

```text
all-MiniLM-L6-v2
```

The model converts text into numerical vectors that capture the semantic meaning of the documents.

## Vector Databases

### FAISS

FAISS is used for efficient similarity search over the generated embeddings.

The project uses `IndexFlatL2` to retrieve the top 3 documents based on vector distance.

### ChromaDB

ChromaDB is used to persist documents and their embeddings locally and perform semantic queries.

The database is stored in:

```text
./chroma_db
```

## Example

**Query:**
```text
How do computers learn from data?
```

**Relevant Results:**
```text
Machine learning allows computers to learn patterns from data.

Deep learning uses neural networks with multiple layers.

Embeddings represent text as numerical vectors.
```

## Installation

```bash
pip install sentence-transformers numpy faiss-cpu chromadb
```

## Run the Project

```bash
python semantic_search.py
```

Enter a query when prompted:

```text
Write a Query:
```

The program returns the **top 3 relevant documents** using both FAISS and ChromaDB.

## Key Learning Outcomes

- Understanding text embeddings
- Understanding semantic search
- Working with vector databases
- Using FAISS for similarity search
- Using ChromaDB for persistent vector storage
- Understanding the foundation of **RAG (Retrieval-Augmented Generation)**

## Future Improvements

- Add a larger document dataset
- Integrate an LLM to generate answers from retrieved documents
- Build a RAG pipeline
- Add a Streamlit user interface
- Compare different embedding models