import chromadb
import ollama
# 1. Connect to DB
client = chromadb.PersistentClient(path=r"vectorstore/chroma_db")
collection = client.get_collection("mitre_attack")
# 2. Embed your query using the SAME model
query_text = "what is filemod in windows start up folder?"
response = ollama.embeddings(model="nomic-embed-text", prompt=query_text)
query_embedding = response["embedding"]
# 3. Search
results = collection.query(
    query_embeddings=[query_embedding],
    n_results=3
)
# 4. Process Results
for i, doc in enumerate(results['documents'][0]):
    print(f"Result {i+1}: {results['metadatas'][0][i]['name']}")
    print(doc)