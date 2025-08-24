# 📚 NCERT RAG Assistant

A Retrieval-Augmented Generation (RAG) project that allows you to ask questions from NCERT textbooks and get context-aware answers using Google Gemini AI + FAISS vector search.

---

## ✨ Features
- Extracts text from NCERT PDFs
- Splits text into semantic chunks & stores embeddings in FAISS
- Supports natural language questions on textbooks
- Uses Google Gemini API for final answers
- CLI-based Q&A interface (Web UI coming soon)

---

## 📂 Project Structure

```
ncert-rag-assistant/
├── data/
│   ├── pdfs/
│   ├── chunks.pkl
│   └── chunks_index.faiss
├── prompts/
│   └── qa_prompt.txt
├── src/
│   ├── build_vector_store.py
│   └── qa_cli.py
├── .env
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup Instructions

### 1. Clone the repository
```bash
git clone https://github.com/your-username/ncert-rag-assistant.git
cd ncert-rag-assistant
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

Example requirements.txt:
```
faiss-cpu==1.8.0
PyPDF2==3.0.1
python-dotenv==1.0.1
google-generativeai==0.7.2
tqdm==4.66.4
# If using pdfplumber instead of PyPDF2:
pdfplumber==0.11.0
```

### 3. Add your API key
Create a `.env` file in the root folder:
```
GEMINI_API_KEY=your_api_key_here
```

### 4. Build the vector store
Put NCERT PDFs in `data/pdfs/` and run:
```bash
python src/build_vector_store.py
```

### 5. Start Q&A
```bash
python src/qa_cli.py
```

---

## 💡 Example Usage
```
❓ Your Question: What is reproduction in animals?
📘 Answer: Reproduction is essential for the continuation of a species...
```

---

## 📌 Notes
- FAISS stores embeddings locally (`chunks_index.faiss`) for fast retrieval.
- CLI interface is lightweight; a web interface can be added later.
- Ensure PDFs are readable and correctly formatted.

---

## 🛠️ Technologies Used
- Python 3.x
- FAISS (Vector Search)
- PyPDF2 / pdfplumber (PDF text extraction)
- Google Gemini AI
- dotenv for environment variables
- Pickle for storing chunks
