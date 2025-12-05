import json
import os
import chromadb
from typing import Set

# Configuration - matches process_mitre.py
MITRE_FILE_PATH = r"datasets/raw/mitre/enterprise-attack.json"
CHROMA_PATH = r"vectorstore/chroma_db"
COLLECTION_NAME = "mitre_attack"
ALLOWED_TYPES = {
    "attack-pattern", 
    "malware", 
    "tool", 
    "intrusion-set",
    "course-of-action",
    "campaign"
}

def main():
    print("--- MITRE Processing Audit ---")
    
    # 1. Analyze Source Data
    print(f"Reading source file: {MITRE_FILE_PATH}")
    if not os.path.exists(MITRE_FILE_PATH):
        print("❌ Source file not found!")
        return

    with open(MITRE_FILE_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)

    all_objects = data.get("objects", [])
    relevant_objects = []
    skipped_types = 0
    skipped_revoked = 0
    
    for obj in all_objects:
        if obj.get("type") not in ALLOWED_TYPES:
            skipped_types += 1
            continue
        
        # Must match filter logic in process_mitre.py
        if obj.get("revoked") or obj.get("x_mitre_deprecated"):
            skipped_revoked += 1
            continue

        relevant_objects.append(obj)
        
    print(f"Total objects in JSON: {len(all_objects)}")
    print(f"Skipped (irrelevant type): {skipped_types}")
    print(f"Skipped (revoked/deprecated): {skipped_revoked}")
    print(f"✅ Expected Embeddings Count: {len(relevant_objects)}")

    # 2. Analyze Vector Store
    print(f"\nConnecting to ChromaDB at: {CHROMA_PATH}")
    try:
        client = chromadb.PersistentClient(path=CHROMA_PATH)
        # Check if collection exists
        try:
            collection = client.get_collection(COLLECTION_NAME)
        except Exception:
            print(f"❌ Collection '{COLLECTION_NAME}' does not exist in ChromaDB.")
            return
            
        db_count = collection.count()
        print(f"✅ Actual Documents in ChromaDB: {db_count}")
        
        diff = len(relevant_objects) - db_count
        
        if diff == 0:
            print("\n🎉 SUCCESS: Database count matches expected count exactly.")
        elif diff > 0:
            print(f"\n⚠️  MISSING: {diff} objects are missing from the database.")
            print("Possible reasons:")
            print("1. Embedding API/Model failures (check process output).")
            print("2. Script was interrupted.")
        else:
            print(f"\n⚠️  EXTRA: Database has {abs(diff)} more items than expected.")
            print("Possible reasons:")
            print("1. Duplicates (script logic adds suffix on ID collision).")
            print("2. Old data was not cleared before run.")

    except Exception as e:
        print(f"❌ Error connecting to ChromaDB: {e}")

if __name__ == "__main__":
    main()
