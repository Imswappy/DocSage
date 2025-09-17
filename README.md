# DocSage — Source-backed, Local RAG Document Q&A
**Ask. Retrieve. Understand.** 🧠

DocSage is a production-style Retrieval-Augmented-Generation (RAG) assistant that ingests arbitrary documents (PDF, DOCX, PPTX, TXT, CSV, images), indexes their semantic embeddings locally, retrieves the most relevant chunks for a question, and generates concise, source-backed answers — all runnable locally (no API keys required).

---

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](#license) [![Python](https://img.shields.io/badge/python-3.8%2B-yellowgreen.svg)](#requirements)

---

## Demo / Elevator pitch
DocSage lets you build a **private, local** knowledge assistant for **any** document collection — legal contracts, research papers, SOPs, manuals, datasheets, support KBs, meeting notes — and ask natural-language questions that return **answers + provenance** (which document & which chunk). Ideal for demos, interviews, and privacy-sensitive workflows.

---

## Key features
- ✅ **Multi-format ingestion**: PDF (text + OCR fallback), DOCX, PPTX, TXT, CSV, PNG/JPG (OCR).
- ✅ **Local embeddings**: `sentence-transformers/all-MiniLM-L6-v2` (fast & small) by default.
- ✅ **Vector store with persistence**: Chroma (duckdb+parquet) with FAISS fallback option.
- ✅ **Local generation**: `google/flan-t5-small` for CPU-friendly generation (swap for larger GPU models).
- ✅ **RAG pipeline**: Retriever → top-k chunk gathering → generation using retrieved context.
- ✅ **Domain-aware prompts**: tweak behavior for `pharma`, `legal`, `engineering`, `general`.
- ✅ **Streamlit UI**: upload, ingest, ask questions, view citations, conversation history.
- ✅ **Backend API (FastAPI)**: `/ingest/` and `/query/` for programmatic integration.
- ✅ **Metadata & tags**: store `domain`, `tags`, `source` per chunk for filtering.
- ✅ **Token estimator & adaptive-K (UI support)**: estimate token usage and optionally let backend pick `k`.
- ✅ **Local-only mode**: no cloud or API keys needed — perfect for private documents.

---

## Project structure
```
pharma-rag/  (root)
├─ backend/
│  ├─ app.py
│  ├─ rag_pipeline.py
│  ├─ requirements.txt
│  └─ .env (optional)
├─ frontend/
│  ├─ streamlit_app.py
│  └─ requirements.txt
├─ data/              # uploaded documents
├─ chroma_db/         # vector db persistence
└─ README.md
```

---

## Quick start (local, VS Code / terminal)
1. Clone repo:
```bash
git clone <your-repo-url>
cd pharma-rag
```

2. Create & activate venv:
```bash
python -m venv .venv
# Windows (PowerShell)
.venv\\Scripts\\Activate.ps1
# macOS / Linux
source .venv/bin/activate
```

3. Install backend deps:
```bash
cd backend
pip install -r requirements.txt
```

4. Start backend:
```bash
python -m uvicorn app:app --reload --port 8000
```

5. Install frontend deps and run UI (new terminal, same venv):
```bash
cd ../frontend
pip install -r requirements.txt
streamlit run streamlit_app.py
```

6. Open Streamlit UI (usually at http://localhost:8501) and FastAPI docs at http://127.0.0.1:8000/docs.

---

## API usage examples

### Ingest a document (curl)
```bash
curl -X POST "http://127.0.0.1:8000/ingest/" \
  -F "file=@/path/to/your/file.pdf" \
  -F "domain=legal" \
  -F "tags=contract,nda"
```

### Query the system (curl)
```bash
curl -X POST "http://127.0.0.1:8000/query/" \
  -H "Content-Type: application/json" \
  -d '{"question":"What termination clauses exist?","top_k":5,"domain":"legal"}'
```

---

## Architecture & dataflow
1. **Ingest**: file → extractor (format-specific), OCR fallback for scanned PDFs/images.  
2. **Chunk**: split text into overlapping chunks (e.g., 800 chars, 150 overlap).  
3. **Embed**: each chunk → vector via SentenceTransformers.  
4. **Index**: store chunks + vectors + metadata in Chroma (persisted).  
5. **Query**: question → embedding → retrieve top-k similar chunks.  
6. **Generate**: join top-k chunks → template prompt → local generator → answer.  
7. **Provenance**: return generated answer + retrieved chunk snippets & metadata.

---

## Models used (default & options)
- **Embeddings (default)**: `sentence-transformers/all-MiniLM-L6-v2` (dim=384).  
- **Generator (default)**: `google/flan-t5-small` (seq2seq, CPU-friendly).  
- **Vector DB**: Chroma; can swap to FAISS or managed providers (Pinecone, Weaviate).  
- **Optional**: OpenAI Embeddings / Chat or Google Gemini via LangChain integrations (if you opt for cloud APIs).

---

## Mathematical & algorithmic intuitions

### Embeddings & similarity
We map text to vectors in ℝᵈ using an embedding model:
\ne\begin{align*}
\mathbf{e} &= \text{Embed}(text) \in \mathbb{R}^d
\end{align*}\n\nCosine similarity between query q and chunk vector v:\n\n\\[\n\\text{sim}(q, v) = \\frac{q \\cdot v}{\\|q\\|\\|v\\|}\n\\]\n\nHigher sim indicates higher relevance.

### Retrieval (top-k)
Select the k most similar chunks to the query embedding. `top_k` trades off context vs cost/latency.

### Chunking rationale
Chunk size and overlap preserve context across boundaries. Defaults: chunk ≈ 800 chars, overlap ≈ 150 chars.

### Maximal Marginal Relevance (MMR) — optional
MMR balances relevance and novelty to reduce redundancy among retrieved chunks:
\\[\n\\text{score}(D_i) = \\lambda \\cdot \\text{sim}(q, D_i) - (1-\\lambda) \\cdot \\max_{D_j \\in S} \\text{sim}(D_j, D_i)\n\\]

### Token estimate & cost (heuristic)
Tokens ≈ characters / 4. Total tokens ≈ avg_chunk_chars × k / 4 + len(question)/4 + overhead.

---

## Evaluation & metrics
- **Retrieval**: Recall@k.  
- **Generation**: human evaluation preferred; automatic metrics (ROUGE/BLEU) are weak.  
- **Provenance accuracy**: how often citations truly support claims (human or automated checks).

---

## Practical tips & heuristics
- Default `top_k`: 3–5.  
- Cache embeddings & skip duplicate ingestion (use file hash).  
- For scanned PDFs, install Tesseract and Poppler.  
- Use MMR or deduplication for noisy results.

---

## Extensibility & production notes
- Swap generator for larger chat models on GPU for better synthesis.  
- Replace Chroma with managed vector DB for scale.  
- Add authentication, background ingestion workers, logging, and monitoring for production.

---

## Security & privacy
- Local-only mode keeps data on your machine — good for PHI/PII.  
- If using cloud APIs, consider redaction and obtain user consent.

---

## Roadmap
- Adaptive-K server-side logic.  
- Table extraction & structured Q&A.  
- Docker images & `docker-compose` for one-command deployment.  
- Fine-tuning generator on domain Q&A pairs.

---

## Contributing
Contributions welcome — fork, branch, PR, include tests & docs.

---

## License
MIT License — see `LICENSE` file.

---

## Acknowledgements
Built with OSS: LangChain, Chroma, SentenceTransformers, Hugging Face Transformers, FastAPI, Streamlit, PyPDF2, pdf2image, pytesseract.

---

## Contact
Open an issue or contact `imswappy.personal@gmail.com` for help or collaboration.
