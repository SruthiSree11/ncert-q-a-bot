# src/build_vectorstore.py
import os
import faiss
import numpy as np
from PyPDF2 import PdfReader
from google import genai
import pickle
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ✅ Get API key from environment
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("GOOGLE_API_KEY not found in environment variables")

# ✅ Initialize client with proper API key
client = genai.Client(api_key=api_key)

INDEX_PATH = "data/chunks_index.faiss"
CHUNKS_PATH = "data/chunks.pkl"

def load_pdfs_from_folder(folder_path):
    """Load all PDFs from a folder and extract text."""
    pdf_texts = []
    if not os.path.exists(folder_path):
        print(f"❌ Folder not found: {folder_path}")
        return pdf_texts
        
    for file in os.listdir(folder_path):
        if file.endswith(".pdf"):
            try:
                reader = PdfReader(os.path.join(folder_path, file))
                text = ""
                for page in reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
                pdf_texts.append((file, text))
                print(f"✓ Loaded: {file} ({len(text)} characters)")
            except Exception as e:
                print(f"❌ Error reading {file}: {e}")
    return pdf_texts

def chunk_text(text, chunk_size=1000, overlap=100):
    """Smart chunking that preserves paragraphs and sentences."""
    chunks = []
    
    # First, split by paragraphs
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    
    for paragraph in paragraphs:
        # If paragraph is small, keep it as-is
        if len(paragraph) <= chunk_size:
            chunks.append(paragraph)
            continue
            
        # If paragraph is large, split by sentences
        import re
        sentences = re.split(r'(?<=[.!?])\s+', paragraph)
        
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            if current_length + len(sentence) > chunk_size and current_chunk:
                chunks.append(' '.join(current_chunk))
                # Keep some overlap
                current_chunk = current_chunk[-3:]  # Keep last 3 sentences
                current_length = sum(len(s) + 1 for s in current_chunk)
            
            current_chunk.append(sentence)
            current_length += len(sentence) + 1
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
    
    return chunks

def get_vector_dim():
    """Detect embedding vector dimension automatically."""
    try:
        response = client.models.embed_content(
            model="models/embedding-001",
            contents=["sample text"]
        )
        vector = response.embeddings[0].values
        return len(vector)
    except Exception as e:
        print(f"❌ Error detecting vector dimension: {e}")
        return 768  # Fallback dimension

def embed_texts(texts):
    """Embed multiple texts efficiently."""
    try:
        response = client.models.embed_content(
            model="models/embedding-001",
            contents=texts
        )
        return np.array([item.values for item in response.embeddings], dtype="float32")
    except Exception as e:
        print(f"❌ Error embedding texts: {e}")
        return np.array([])

def build_faiss_index(folder_path):
    """Build FAISS index from PDFs in folder."""
    # Create data directory if it doesn't exist
    os.makedirs("data", exist_ok=True)
    
    print("📖 Loading PDFs...")
    pdf_texts = load_pdfs_from_folder(folder_path)
    
    if not pdf_texts:
        print("❌ No PDFs found or loaded")
        return None, None

    all_chunks = []
    all_embeddings_list = []

    # Auto-detect embedding dimension
    print("🔍 Detecting embedding dimension...")
    VECTOR_DIM = get_vector_dim()
    print(f"📊 Vector dimension: {VECTOR_DIM}")

    for file, text in pdf_texts:
        print(f"📄 Processing: {file}")
        chunks = chunk_text(text)
        print(f"   → Split into {len(chunks)} chunks")
        
        if chunks:
            # Process chunks in batches to avoid API limits
            batch_size = 10
            for i in range(0, len(chunks), batch_size):
                batch = chunks[i:i + batch_size]
                embeddings = embed_texts(batch)
                
                if len(embeddings) > 0:
                    all_embeddings_list.append(embeddings)
                    all_chunks.extend(batch)
                    print(f"   → Embedded batch {i//batch_size + 1}")

    if not all_embeddings_list:
        print("❌ No embeddings generated")
        return None, None

    # Combine all embeddings
    all_embeddings = np.vstack(all_embeddings_list)

    # Build FAISS index (using Inner Product for cosine similarity)
    index = faiss.IndexFlatIP(VECTOR_DIM)
    
    # Normalize vectors for cosine similarity
    faiss.normalize_L2(all_embeddings)
    index.add(all_embeddings)
    faiss.normalize_L2(all_embeddings)  # ← ADD THIS LINE
    index.add(all_embeddings)
    # Save FAISS index + chunks
    faiss.write_index(index, INDEX_PATH)
    with open(CHUNKS_PATH, "wb") as f:
        pickle.dump(all_chunks, f)

    print(f"✅ Vectorstore built successfully!")
    print(f"   - Chunks: {len(all_chunks)}")
    print(f"   - Dimension: {VECTOR_DIM}")
    print(f"   - Index saved: {INDEX_PATH}")
    print(f"   - Chunks saved: {CHUNKS_PATH}")
    
    return all_chunks, VECTOR_DIM

if __name__ == "__main__":
    folder = "data/pdfs"  # Path to your PDFs folder
    
    # Check if folder exists
    if not os.path.exists(folder):
        print(f"❌ PDF folder not found: {folder}")
        print("💡 Create a 'data/pdfs' folder and add your NCERT PDFs there")
    else:
        build_faiss_index(folder)
