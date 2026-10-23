from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import chromadb


# ==========================================
# 1. DOCUMENTS
# ==========================================

documents = [
    "Machine learning allows computers to learn patterns from data.",
    "Deep learning uses neural networks with multiple layers.",
    "Natural Language Processing enables computers to understand human language.",
    "Transformers are neural network architectures widely used in NLP.",
    "Embeddings represent text as numerical vectors.",
    "Vector databases store and search high-dimensional vectors.",
    "FAISS is a library for efficient similarity search.",
    "ChromaDB is a vector database used for storing embeddings.",
    "RAG combines document retrieval with large language models.",
    "Cosine similarity measures how similar two vectors are."
]


# ==========================================
# 2. LOAD EMBEDDING MODEL
# ==========================================

model = SentenceTransformer("all-MiniLM-L6-v2")

print("Embedding model loaded.")


# ==========================================
# 3. GENERATE EMBEDDINGS
# ==========================================

embeddings = model.encode(documents)

embeddings = np.array(embeddings).astype("float32")

print("Embedding shape:", embeddings.shape)


# ==========================================
# 4. FAISS VECTOR DATABASE
# ==========================================

dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)

index.add(embeddings)

print("Number of vectors in FAISS:", index.ntotal)


# ==========================================
# 5. SEMANTIC SEARCH USING FAISS
# ==========================================

query = input("Write a Query:")

query_embedding = model.encode([query])

query_embedding = np.array(query_embedding).astype("float32")

distances, indices = index.search(query_embedding, k=3)


print("\n========== FAISS SEARCH ==========")

print("Query:", query)

for i, idx in enumerate(indices[0]):
    print(f"\nResult {i + 1}")
    print("Document:", documents[idx])
    print("Distance:", distances[0][i])


# ==========================================
# 6. CHROMADB
# ==========================================

client = chromadb.PersistentClient(
    path="./chroma_db"
)

collection = client.get_or_create_collection(
    name="semantic_documents"
)


# Add documents to ChromaDB

collection.add(
    ids=[f"doc_{i}" for i in range(len(documents))],
    documents=documents,
    embeddings=embeddings.tolist()
)

print("\nDocuments stored in ChromaDB.")


# ==========================================
# 7. SEMANTIC SEARCH USING CHROMADB
# ==========================================

results = collection.query(
    query_embeddings=query_embedding.tolist(),
    n_results=3
)


print("\n========== CHROMADB SEARCH ==========")

print("Query:", query)

for i, doc in enumerate(results["documents"][0]):
    print(f"\nResult {i + 1}")
    print("Document:", doc)