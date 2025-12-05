import chromadb
import ollama
import sys
import argparse

# Configuration
CHROMA_PATH = r"vectorstore/chroma_db"
COLLECTION_NAME = "mitre_attack"
EMBEDDING_MODEL = "nomic-embed-text"
LLM_MODEL = "DeepHat/DeepHat-V1-7B"

def get_context(query: str, collection, n_results: int = 3) -> str:
    """Retrieves relevant context from ChromaDB."""
    try:
        # Generate embedding for the query
        response = ollama.embeddings(model=EMBEDDING_MODEL, prompt=query)
        query_embedding = response["embedding"]
        
        # Query ChromaDB
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results
        )
        
        # Format context
        context_parts = []
        if results and results['documents']:
            for i, doc in enumerate(results['documents'][0]):
                meta = results['metadatas'][0][i]
                source = f"Source: {meta.get('mitre_id')} ({meta.get('name')})"
                context_parts.append(f"{source}\n{doc}")
                
        return "\n\n---\n\n".join(context_parts)
    except Exception as e:
        print(f"Error retrieving context: {e}")
        return ""

def chat_loop():
    print("⏳ Connecting to Vector Store...")
    try:
        client = chromadb.PersistentClient(path=CHROMA_PATH)
        collection = client.get_collection(COLLECTION_NAME)
        print(f"✅ Connected to collection: {COLLECTION_NAME}")
    except Exception as e:
        print(f"❌ Failed to connect to ChromaDB: {e}")
        sys.exit(1)

    print(f"🤖 AI-SOC Assistant ({LLM_MODEL}) ready.")
    print("Type 'exit' or 'quit' to stop.\n")

    history = []

    while True:
        try:
            user_input = input("User: ").strip()
            if user_input.lower() in ['exit', 'quit']:
                break
            if not user_input:
                continue

            # 1. Retrieve Context
            print("🔍 Searching knowledge base...", end='\r')
            context = get_context(user_input, collection)
            print("                           ", end='\r') # clear line

            # 2. Construct System Prompt
            system_prompt = (
                "You are an expert AI SOC Analyst assistant. "
                "Use the provided MITRE ATT&CK context to answer the user's question accurately. "
                "If the answer is not in the context, use your general knowledge but mention that it's not in the specific retrieved data.\n\n"
                f"CONTEXT:\n{context}"
            )

            # 3. Call LLM
            messages = [
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': user_input}
            ]
            
            print("🤔 Thinking...", end='\r')
            stream = ollama.chat(
                model=LLM_MODEL,
                messages=messages,
                stream=True
            )
            
            print("Assistant: ", end='', flush=True)
            full_response = ""
            for chunk in stream:
                content = chunk['message']['content']
                print(content, end='', flush=True)
                full_response += content
            print("\n")
            
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")

if __name__ == "__main__":
    chat_loop()
