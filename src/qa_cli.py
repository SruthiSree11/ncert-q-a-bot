# src/qa_cli.py
import faiss
import pickle
import numpy as np
from google import genai
import os

# ✅ Initialize Gemini client with Google GenAI (consistent with build_vectorstore)
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    print("❌ Error: GOOGLE_API_KEY environment variable is not set!")
    exit(1)

try:
    client = genai.Client(api_key=api_key)
    print("✅ Successfully connected to Gemini API")
except Exception as e:
    print(f"❌ Error initializing Gemini client: {e}")
    exit(1)

# --- Load FAISS index & chunks ---
print("📂 Loading FAISS index & chunks...")
try:
    index = faiss.read_index("data/chunks_index.faiss")
    with open("data/chunks.pkl", "rb") as f:
        chunks = pickle.load(f)
    print(f"✅ Loaded {len(chunks)} chunks")
except FileNotFoundError:
    print("❌ Error: Vector store files not found. Run build_vectorstore.py first!")
    exit(1)

# --- Embed query using Google GenAI ---
def embed_text(text):
    """Embed text using Gemini's embedding API with Google GenAI"""
    try:
        response = client.models.embed_content(
            model="models/embedding-001",
            contents=[text]
        )
        embedding = np.array(response.embeddings[0].values, dtype="float32")
        return embedding / np.linalg.norm(embedding)  # ✅ Normalize for cosine similarity
    except Exception as e:
        print(f"❌ Error embedding text: {e}")
        return None

# --- Retrieve relevant chunks ---
def retrieve(query, top_k=5):
    query_vec = embed_text(query)
    if query_vec is None:
        return []
    
    query_vec = query_vec.reshape(1, -1)
    scores, indices = index.search(query_vec, top_k)
    
    retrieved = []
    for rank, (idx, score) in enumerate(zip(indices[0], scores[0]), start=1):
        if 0 <= idx < len(chunks):
            retrieved.append({
                "content": chunks[idx],
                "score": float(score),
                "rank": rank
            })
    
    return retrieved

# --- Ask Gemini with context ---
def ask_gemini(prompt):
    """Ask Gemini using Google GenAI client"""
    try:
        response = client.models.generate_content(
            model="gemini-1.5-flash",
            contents=prompt
        )
        return response.text
    except Exception as e:
        return f"❌ Error: {str(e)}"

def ask(query):
    retrieved_chunks = retrieve(query, top_k=3)
    print(f"🔍 Retrieved {len(retrieved_chunks)} chunks for query: '{query}'")
    
    # Format context
    context_parts = []
    for i, chunk in enumerate(retrieved_chunks):
        print(f"   Chunk {i+1}: score={chunk['score']:.4f}, preview: {str(chunk['content'])[:50]}...")
        context_parts.append(f"[Source {i+1}, Relevance: {chunk['score']:.3f}]\n{chunk['content']}")
    
    context = "\n\n".join(context_parts)

    prompt = f"""
    You are an NCERT assistant. Use the following textbook excerpts to answer the question clearly and factually.

    QUESTION: {query}

    CONTEXT FROM NCERT TEXTBOOK:
    {context}

    INSTRUCTIONS:
    - Answer based only on the provided context
    - If the context doesn't contain the answer, say "I don't have enough information from the NCERT textbook to answer this question"
    - Keep your answer clear and concise
    - Use simple language appropriate for students

    ANSWER:
    """

    return ask_gemini(prompt)

# --- CLI loop ---
if __name__ == "__main__":
    print("\n💡 NCERT RAG Assistant Ready!")
    print("📚 Using Gemini AI with your NCERT textbook content")
    print("🔎 Type 'exit' or 'quit' to end the session\n")
    
    while True:
        try:
            query = input("❓ Your Question: ").strip()
            if query.lower() in ["exit", "quit", ""]:
                print("👋 Goodbye!")
                break
                
            if not query:
                continue
                
            print("🤔 Thinking...")
            answer = ask(query)
            print("\n📘 Answer:", answer, "\n")
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
