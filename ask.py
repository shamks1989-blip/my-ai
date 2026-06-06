from sentence_transformers import SentenceTransformer
import chromadb, requests

embedder = SentenceTransformer("all-MiniLM-L6-v2")
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_collection("hr_docs")

def get_answer(question):
    vec = embedder.encode(question).tolist()
    results = collection.query(
        query_embeddings=[vec], n_results=2)
    context = "\n\n".join(results["documents"][0])

    prompt = f"""You are an HR assistant.
Answer using ONLY the context below.
If not found say: I don't have that information.

Context:
{context}

Question: {question}
Answer:"""

    r = requests.post("http://localhost:11434/api/generate",
        json={"model":"llama3","prompt":prompt,"stream":False})
    data = r.json()
    if "response" not in data:
        raise RuntimeError(f"Ollama error: {data.get('error', data)}")
    return data["response"]

print("HR Bot ready! Type exit to quit.\n")
while True:
    q = input("You: ").strip()
    if q.lower() in ["exit","quit"]: break
    if q: print(f"\nBot: {get_answer(q)}\n")