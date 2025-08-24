## 📌 Step 1: `requirements.txt`

Since you’re using:

* **FAISS** for vector search
* **PyPDF2 / pdfplumber** (depending on what you used for text extraction)
* **dotenv** for API key loading
* **Google Generative AI (Gemini)** SDK
* **Pickle** (built-in, no need to install)

Here’s a safe minimal file:

```txt
faiss-cpu==1.8.0
PyPDF2==3.0.1
python-dotenv==1.0.1
google-generativeai==0.7.2
tqdm==4.66.4
```

👉 If you also used `pdfplumber` instead of `PyPDF2`, add:

```txt
pdfplumber==0.11.0
```

---

## 📌 Step 2: `README.md`

Here’s a clean version you can drop in root:

```markdown
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
```

.
├── data
│   ├── pdfs/                 # NCERT PDFs
│   ├── chunks.pkl            # Stored text chunks
│   └── chunks\_index.faiss    # FAISS index
├── prompts
│   └── qa\_prompt.txt         # Prompt template for Gemini
├── src
│   ├── build\_vector\_store.py # PDF → Chunks → Embeddings → FAISS
│   └── qa\_cli.py             # CLI chatbot for Q\&A
├── .env                      # Gemini API key here
├── requirements.txt
└── README.md

````

---

## ⚙️ Setup Instructions

### 1. Clone repo
```bash
git clone https://github.com/your-username/ncert-rag-assistant.git
cd ncert-rag-assistant
````

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Add your API key

Create a `.env` file:

```
GEMINI_API_KEY=your_api_key_here
```

### 4. Build the vector store

Put NCERT PDFs in `data/pdfs/`, then run:

```bash
python src/build_vector_store.py
```

### 5. Start Q\&A

```bash
python src/qa_cli.py
```

---

## 💡 Example Usage

```
❓ Your Question: What is reproduction in animals?
📘 Answer: Reproduction is essential for the continuation of a species...
```






