from sentence_transformers import SentenceTransformer
import chromadb

print("Loading model...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

client = chromadb.PersistentClient(path="./chroma_db")
try:
    client.delete_collection("hr_docs")
except:
    pass
collection = client.create_collection("hr_docs")

with open("hr_data.txt", "r") as f:
    text = f.read()

chunks = [p.strip() for p in text.split("\n\n") if p.strip()]
print(f"Found {len(chunks)} chunks")

embeddings = embedder.encode(chunks).tolist()

collection.add(
    documents=chunks,
    embeddings=embeddings,
    ids=[f"chunk_{i}" for i in range(len(chunks))]
)
print(f"Done! Stored {len(chunks)} chunks. Run ask.py next.")