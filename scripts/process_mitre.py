import json
import os
import chromadb
import ollama
from typing import List, Dict, Any, Optional

# Configuration
MITRE_FILE_PATH = r"datasets/raw/mitre/enterprise-attack.json"
CHROMA_PATH = r"vectorstore/chroma_db"
COLLECTION_NAME = "mitre_attack"
# Expanded types based on "process data as you see fit"
ALLOWED_TYPES = {
    "attack-pattern", 
    "malware", 
    "tool", 
    "intrusion-set",
    "course-of-action", # Mitigation
    "campaign"
}
OLLAMA_MODEL = "nomic-embed-text"

def load_mitre_data(file_path: str) -> List[Dict[str, Any]]:
    """Loads and filters MITRE ATT&CK data from the JSON file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"MITRE file not found at: {file_path}")

    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    processed_objects = []
    
    for obj in data.get("objects", []):
        if obj.get("type") not in ALLOWED_TYPES:
            continue
        
        # Revoked or deprecated objects
        if obj.get("revoked") or obj.get("x_mitre_deprecated"):
            continue

        processed_objects.append(obj)
        
    return processed_objects

def extract_mitre_id(external_references: List[Dict[str, str]]) -> str:
    """Extracts the MITRE ATT&CK ID (e.g., T1001) from external references."""
    for ref in external_references:
        if ref.get("source_name") == "mitre-attack":
            return ref.get("external_id", "UNKNOWN")
    return "UNKNOWN"

def normalize_list(item: Any) -> str:
    """Normalizes a list into a comma-separated string."""
    if isinstance(item, list):
        return ", ".join(str(i) for i in item)
    return str(item) if item is not None else ""

def generate_document_text(obj: Dict[str, Any], mitre_id: str) -> str:
    """Generates a rich unified text representation for embedding."""
    name = obj.get("name", "Unknown")
    description = obj.get("description", "")
    obj_type = obj.get("type")
    
    # Context-specific fields
    platforms = normalize_list(obj.get("x_mitre_platforms"))
    detection = obj.get("x_mitre_detection", "")
    permissions = normalize_list(obj.get("x_mitre_permissions_required"))
    
    # Construct a comprehensive text document
    # Nomic embedding models perform well with semi-structured text.
    header = f"{mitre_id}: {name} ({obj_type})"
    
    text_parts = [
        header,
        "=" * len(header),
        f"Description: {description}",
    ]
    
    if platforms:
        text_parts.append(f"Platforms: {platforms}")
    if permissions:
        text_parts.append(f"Permissions Required: {permissions}")
    if detection:
        text_parts.append(f"Detection: {detection}")
        
    # Append distinct tags/keywords if available can help, but raw text is usually sufficient.
    
    return "\n\n".join(text_parts)

def get_embedding(text: str) -> Optional[List[float]]:
    """Generates an embedding using local Ollama instance."""
    try:
        response = ollama.embeddings(model=OLLAMA_MODEL, prompt=text)
        return response["embedding"]
    except Exception as e:
        print(f"Error generating embedding with Ollama: {e}")
        return None

def main():
    print(f"Loading data from {MITRE_FILE_PATH}...")
    objects = load_mitre_data(MITRE_FILE_PATH)
    print(f"Found {len(objects)} relevant objects.")

    documents = []
    metadatas = []
    ids = []
    embeddings = []

    print("Processing objects and generating embeddings...")
    # Process in batches or one by one. Ollama can be fast for single embeddings but blocking.
    # We will do it sequentially with a simple progress indicator.
    
    count = 0
    total = len(objects)
    
    for obj in objects:
        count += 1
        mitre_id = extract_mitre_id(obj.get("external_references", []))
        if mitre_id == "UNKNOWN":
            mitre_id = obj.get("id")

        name = obj.get("name", "Unknown")
        obj_type = obj.get("type")
        
        # Build Metadata
        metadata = {
            "mitre_id": mitre_id,
            "name": name,
            "type": obj_type,
            "url": next((ref.get("url", "") for ref in obj.get("external_references", []) if ref.get("source_name") == "mitre-attack"), ""),
            "platforms": normalize_list(obj.get("x_mitre_platforms")),
            "deprecated": str(obj.get("x_mitre_deprecated", False))
        }
        
        # Build Document Text
        text_content = generate_document_text(obj, mitre_id)
        
        # Generate Embedding
        if count % 10 == 0:
            print(f"Processing {count}/{total}...", end='\r')
            
        emb = get_embedding(text_content)
        if emb is None:
            # Skip or retry? We skip for now.
            print(f"\nSkipping {mitre_id} due to embedding failure.")
            continue

        documents.append(text_content)
        metadatas.append(metadata)
        # Unique ID for Chroma: MITRE ID is good, but if there are dupes (unlikely for active), use obj['id']
        # We will use obj['id'] (STIX ID) to be absolutely safe against collisions, but store MITRE ID in metadata.
        # Actually user requested "ids = MITRE technique IDs". We will try that.
        # If collision, we might lose data. Let's append type if needed to make unique? 
        # T-codes are unique for techniques. S-codes for software. 
        # But let's check duplicates while processing.
        uniq_id = mitre_id
        if uniq_id in ids:
            # simple dedupe strategy
            uniq_id = f"{mitre_id}_{obj.get('id')}"
            
        ids.append(uniq_id)
        embeddings.append(emb)

    print(f"\nGenerated {len(embeddings)} embeddings.")

    print(f"Initializing ChromaDB at {CHROMA_PATH}...")
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_or_create_collection(name=COLLECTION_NAME)

    print(f"Upserting to ChromaDB...")
    batch_size = 100
    for i in range(0, len(documents), batch_size):
        end = min(i + batch_size, len(documents))
        print(f"Upserting batch {i} to {end}...")
        collection.upsert(
            documents=documents[i:end],
            embeddings=embeddings[i:end],
            metadatas=metadatas[i:end],
            ids=ids[i:end]
        )

    print("Done!")

if __name__ == "__main__":
    main()
