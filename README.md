# 📚 NCERT RAG Assistant

A Retrieval-Augmented Generation (RAG) project that allows you to **ask questions from NCERT textbooks** and get accurate, context-aware answers using **Gemini AI + FAISS vector search**.

---

## ✨ Features
- Extracts text from NCERT PDFs
- Splits into semantic chunks & stores embeddings in FAISS
- Asks natural language questions from textbooks
- Uses **Google Gemini API** for final answers
- CLI-based Q&A interface (web app planned)

---

## 📂 Project Structure
├── data
│ ├── pdfs/ # NCERT PDFs
│ ├── chunks.pkl # Stored text chunks
│ └── chunks_index.faiss # FAISS index
├── prompts
│ └── qa_prompt.txt # Prompt template for Gemini
├── src
│ ├── build_vector_store.py # PDF → Chunks → Embeddings → FAISS
│ └── qa_cli.py # CLI chatbot for Q&A
├── .env # Gemini API key here
├── requirements.txt
└── README.md


---

## ⚙️ Setup Instructions

### 1. Clone repo
```bash
git clone https://github.com/your-username/ncert-rag-assistant.git
cd ncert-rag-assistant

2. Install dependencies

pip install -r requirements.txt

3. Add your API key

Create a .env file:

GEMINI_API_KEY=your_api_key_here

4. Build the vector store

Put NCERT PDFs in data/pdfs/, then run:

python src/build_vector_store.py

5. Start Q&A
python src/qa_cli.py

💡 Example Usage
❓ Your Question: What is reproduction in animals?
📘 Answer: Reproduction is essential for the continuation of a species...