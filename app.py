from sentence_transformers import SentenceTransformer
import chromadb, requests, gradio as gr

embedder = SentenceTransformer("all-MiniLM-L6-v2")
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_collection("hr_docs")

def get_answer(question, history):
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
    return r.json()["response"]

gr.ChatInterface(
    fn=get_answer,
    title="HR Assistant Bot",
    description="Ask me anything about company HR policies",
    examples=["How many leave days?","When is salary credited?",
              "What is notice period?","Can I work from home?"]
).launch()