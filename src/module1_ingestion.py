# Module 1: Sliding Window Chunking with Semantic Boundaries
import os
import glob

def create_ingestion_pipeline(data_path="./data/"):
    """
    Sliding Window Chunking with semantic boundary detection.
    
    Algorithm:
    - Loads all .txt/.md files from data_path
    - Uses sliding window (size=5000, overlap=500)
    - Detects semantic boundaries (paragraphs, sentences) for clean cuts
    - Returns list of chunks with metadata for RAG retrieval
    
    Returns: List of {"text": chunk_content, "source": filename, "index": chunk_id}
    """
    chunks = []
    patterns = ['*.txt', '*.md', '**/*.txt', '**/*.md']
    chunk_id = 0
    
    for pattern in patterns:
        file_paths = glob.glob(os.path.join(data_path, pattern), recursive=True)
        
        for file_path in file_paths:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
                    
                # Sliding window with semantic boundary detection
                chunk_size = 5000
                overlap = 500
                step = chunk_size - overlap
                
                for i in range(0, len(text), step):
                    chunk = text[i : i + chunk_size]
                    
                    # Find semantic boundary (paragraph or sentence end near chunk boundary)
                    if len(chunk) == chunk_size and i + chunk_size < len(text):
                        # Look for paragraph break within last 200 chars
                        para_match = chunk.rfind("\n\n")
                        if para_match > chunk_size - 200:
                            chunk = chunk[:para_match].strip()
                        else:
                            # Fallback: find sentence end
                            sent_match = max(
                                chunk.rfind("."),
                                chunk.rfind("!"),
                                chunk.rfind("?")
                            )
                            if sent_match > chunk_size - 200 and sent_match > 0:
                                chunk = chunk[:sent_match + 1].strip()
                    
                    if len(chunk.strip()) > 100:  # Skip very small chunks
                        chunks.append({
                            'text': chunk,
                            'source': os.path.basename(file_path),
                            'index': chunk_id
                        })
                        chunk_id += 1
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
    
    print(f"Loaded {len(chunks)} chunks from {data_path}")
    return chunks

# --- Testing the Module ---
if __name__ == "__main__":
    # Simple local test without external dependencies
    import os
    os.makedirs("./data", exist_ok=True)
    with open("./data/test_novel.txt", "w", encoding="utf-8") as f:
        f.write("This is a sample novel text for testing ingestion.\n\nParagraph two with more content.")

    pipeline = create_ingestion_pipeline("./data/")
    print(f"Indexed {len(pipeline)} chunks in test mode.")