import os
import re
import shutil
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    StorageContext,
    load_index_from_storage,
    Settings,
)

from llama_index.llms.gemini import Gemini
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.prompts import PromptTemplate


# =========================
# ENV
# =========================

load_dotenv()

UPLOAD_DIR = "uploads"
STORAGE_DIR = "storage"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(STORAGE_DIR, exist_ok=True)

ALLOWED_EXTENSIONS = {".pdf", ".txt", ".docx"}


# =========================
# FAST STARTUP → LAZY AI INIT
# =========================

_ai_initialized = False


def init_ai():
    global _ai_initialized

    if _ai_initialized:
        return

    print("🚀 Initializing AI models...")

    Settings.llm = Gemini(
        model="models/gemini-2.5-flash",
        temperature=0.1,
    )

    # Settings.embed_model = HuggingFaceEmbedding(
    #     model_name="sentence-transformers/paraphrase-MPNet-base-v2"
    # )

    Settings.embed_model = HuggingFaceEmbedding(
        model_name="sentence-transformers/paraphrase-MiniLM-L3-v2"
    )

    _ai_initialized = True
    print("✅ AI initialized")


# =========================
# STRICT LEGAL PROMPT
# =========================

LEGAL_QA_PROMPT = PromptTemplate(
    """
You are a legal document analysis assistant.

STRICT RULES:
- Answer ONLY using the provided document context
- Do NOT use outside knowledge
- Do NOT guess or infer
- If the answer is not explicitly stated, respond exactly with:
  "The document does not contain this information."
- Do NOT provide legal advice or opinions
- Keep answers factual and concise
- Cite page numbers when possible

Context:
{context_str}

Question:
{query_str}

Answer:
"""
)

# =========================
# FASTAPI
# =========================

app = FastAPI(title="LawChatAI – Secure RAG Service")


# =========================
# CORS
# =========================

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "https://lawchatai.in",
        "*",
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# =========================
# HELPERS
# =========================

def validate_user_id(user_id: str):
    if not re.match(r"^[a-zA-Z0-9_-]{3,50}$", user_id):
        raise ValueError("Invalid user_id")


def get_user_dirs(user_id: str):
    user_upload_dir = os.path.join(UPLOAD_DIR, user_id)
    user_storage_dir = os.path.join(STORAGE_DIR, user_id)

    os.makedirs(user_upload_dir, exist_ok=True)
    os.makedirs(user_storage_dir, exist_ok=True)

    return user_upload_dir, user_storage_dir


def build_or_load_index(user_id: str):
    _, user_storage_dir = get_user_dirs(user_id)

    if os.path.exists(os.path.join(user_storage_dir, "docstore.json")):
        storage_context = StorageContext.from_defaults(
            persist_dir=user_storage_dir
        )
        return load_index_from_storage(storage_context)

    raise ValueError("Index does not exist")


# =========================
# API: UPLOAD
# =========================

@app.post("/upload")
async def upload_document(
    user_id: str = Form(...),
    file: UploadFile = File(...),
):
    try:
        validate_user_id(user_id)
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=400)

    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        return JSONResponse(
            {"error": "Unsupported file type"},
            status_code=400,
        )

    # 🔥 init AI here, not at startup
    init_ai()

    user_upload_dir, user_storage_dir = get_user_dirs(user_id)
    file_path = os.path.join(user_upload_dir, file.filename)

    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    documents = SimpleDirectoryReader(user_upload_dir).load_data()

    index = VectorStoreIndex.from_documents(
        documents,
        show_progress=True,
    )

    index.storage_context.persist(persist_dir=user_storage_dir)

    return {
        "status": "success",
        "message": f"{file.filename} uploaded and indexed",
    }


# =========================
# API: QUERY
# =========================

@app.post("/query")
async def query_documents(
    user_id: str = Form(...),
    question: str = Form(...),
):
    try:
        validate_user_id(user_id)
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=400)

    # 🔥 init AI here, not at startup
    init_ai()

    try:
        index = build_or_load_index(user_id)
    except ValueError:
        return JSONResponse(
            {"error": "No documents indexed for this user"},
            status_code=400,
        )

    query_engine = index.as_query_engine(
        similarity_top_k=4,
        text_qa_template=LEGAL_QA_PROMPT,
    )

    response = query_engine.query(question)

    citations = []
    for node in response.source_nodes:
        meta = node.node.metadata
        page = meta.get("page_label") or meta.get("page") or "N/A"

        citations.append({
            "page": page,
            "score": round(node.score, 3),
        })

    return {
        "question": question,
        "answer": response.response,
        "citations": citations,
    }


# =========================
# HEALTH
# =========================

@app.get("/")
def health_check():
    return {"status": "LawChatAI RAG system running"}
