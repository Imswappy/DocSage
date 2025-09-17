import os
import uuid

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from rag_pipeline import answer_query_local, build_retriever, ingest_document

load_dotenv()

DATA_DIR = os.getenv("DATA_DIR", "../data")
os.makedirs(DATA_DIR, exist_ok=True)

app = FastAPI(title="Local Docs RAG Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # dev only
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    question: str
    top_k: int = 5
    adaptive_k: bool = False
    domain: str | None = None

@app.post("/ingest/")
async def ingest(file: UploadFile = File(...), domain: str = Form(None), tags: str = Form(None)):
    contents = await file.read()
    filename = os.path.join(DATA_DIR, file.filename)
    with open(filename, "wb") as f:
        f.write(contents)
    doc_id = str(uuid.uuid4())[:8]
    n_chunks = ingest_document(filename, doc_id=doc_id, domain=domain, tags=tags)
    return {"status":"ok", "doc_id": doc_id, "filename": file.filename, "chunks": n_chunks}

@app.post("/query/")
async def query(req: QueryRequest):
    # For now, adaptive_k is not implemented server-side (placeholder).
    retriever = build_retriever(k=req.top_k, domain=req.domain)
    res = answer_query_local(req.question, retriever, top_k=req.top_k, domain=req.domain)
    return res
