import os
import re
import shutil
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import tempfile

from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    StorageContext,
    load_index_from_storage,
    Settings,
)

from llama_index.llms.gemini import Gemini
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from llama_index.core.prompts import PromptTemplate
from llama_index.core.node_parser import SentenceSplitter

import hashlib
import boto3
from botocore.client import Config


# =========================
# ENV
# =========================

load_dotenv()


# =========================
# R2 CONFIG
# =========================

R2_ACCOUNT_ID = os.getenv("R2_ACCOUNT_ID")
R2_ACCESS_KEY = os.getenv("R2_ACCESS_KEY_ID")
R2_SECRET_KEY = os.getenv("R2_SECRET_ACCESS_KEY")
R2_BUCKET = os.getenv("R2_BUCKET_NAME")

s3 = boto3.client(
    "s3",
    endpoint_url=f"https://{R2_ACCOUNT_ID}.r2.cloudflarestorage.com",
    aws_access_key_id=R2_ACCESS_KEY,
    aws_secret_access_key=R2_SECRET_KEY,
    config=Config(signature_version="s3v4"),
)


ALLOWED_EXTENSIONS = {".pdf", ".txt", ".docx"}


# =========================
# GLOBAL LLAMA SETTINGS (MVP FAST)
# =========================

Settings.chunk_size = 512
Settings.chunk_overlap = 50

Settings.llm = Gemini(
    model="models/gemini-2.5-flash",
    temperature=0.1,
)

Settings.embed_model = GoogleGenAIEmbedding(
    model_name="gemini-embedding-001",
    output_dimensionality=768
)

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
- Cite page numbers when available

Context:
{context_str}

Question:
{query_str}

Answer:
"""
)

def get_file_hash(file_bytes: bytes):
    return hashlib.sha256(file_bytes).hexdigest()

# =========================
# APP
# =========================

app = FastAPI(title="LawChatAI – MVP RAG")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten later
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


def load_user_index_from_r2(user_id: str):
    index_prefix = f"users/{user_id}/index/"
    # local_index_dir = f"/tmp/index_{user_id}"
    local_index_dir = os.path.join(tempfile.gettempdir(), f"index_{user_id}")

    os.makedirs(local_index_dir, exist_ok=True)

    objs = s3.list_objects_v2(Bucket=R2_BUCKET, Prefix=index_prefix)

    if "Contents" not in objs:
        return None

    for obj in objs["Contents"]:
        key = obj["Key"]
        local_file = os.path.join(local_index_dir, key.split("/")[-1])
        s3.download_file(R2_BUCKET, key, local_file)

    storage_context = StorageContext.from_defaults(
        persist_dir=local_index_dir
    )

    return load_index_from_storage(storage_context)



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
        return JSONResponse({"error": "Unsupported file type"}, status_code=400)

    # read file
    file_bytes = await file.read()
    file_hash = get_file_hash(file_bytes)

    # R2 paths
    file_key = f"users/{user_id}/files/{file_hash}_{file.filename}"
    index_prefix = f"users/{user_id}/index/"

    # check if already exists
    try:
        s3.head_object(Bucket=R2_BUCKET, Key=file_key)
        return {
            "status": "success",
            "message": "File already uploaded. Skipping re-index."
        }
    except:
        pass  # not exists, continue

    # upload file
    s3.put_object(
        Bucket=R2_BUCKET,
        Key=file_key,
        Body=file_bytes,
    )

    # we must download locally temporarily for llamaindex
    # temp_path = f"/tmp/{file_hash}_{file.filename}"

    temp_dir = tempfile.gettempdir()
    temp_path = os.path.join(temp_dir, f"{file_hash}_{file.filename}")

    with open(temp_path, "wb") as f:
        f.write(file_bytes)

    documents = SimpleDirectoryReader(input_files=[temp_path]).load_data()

    # download existing index if exists
    local_index_dir = f"/tmp/index_{user_id}"
    os.makedirs(local_index_dir, exist_ok=True)

    try:
        objs = s3.list_objects_v2(Bucket=R2_BUCKET, Prefix=index_prefix)
        for obj in objs.get("Contents", []):
            key = obj["Key"]
            local_file = os.path.join(local_index_dir, key.split("/")[-1])
            s3.download_file(R2_BUCKET, key, local_file)

        storage_context = StorageContext.from_defaults(
            persist_dir=local_index_dir
        )
        index = load_index_from_storage(storage_context)

        parser = SentenceSplitter()

        nodes = parser.get_nodes_from_documents(documents)

        index.insert_nodes(nodes)

    except:
        index = VectorStoreIndex.from_documents(documents)

    # persist locally
    index.storage_context.persist(persist_dir=local_index_dir)

    # upload index back to R2
    for fname in os.listdir(local_index_dir):
        s3.upload_file(
            os.path.join(local_index_dir, fname),
            R2_BUCKET,
            f"{index_prefix}{fname}",
        )

    return {
        "status": "success",
        "message": "Uploaded & indexed"
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
        index = load_user_index_from_r2(user_id)

        if not index:
            raise ValueError()

    except ValueError:
        return JSONResponse(
            {"error": "No documents indexed"},
            status_code=400,
        )

    query_engine = index.as_query_engine(
        similarity_top_k=3,  # faster
        text_qa_template=LEGAL_QA_PROMPT,
    )

    response = query_engine.query(question)

    citations = []
    for node in response.source_nodes:
        meta = node.node.metadata
        citations.append({
            "file": meta.get("file_name", "unknown"),
            "page": meta.get("page_label") or meta.get("page") or "N/A"
        })

    return {
        "answer": response.response,
        "citations": citations,
    }


# =========================
# HEALTH
# =========================

@app.get("/")
def health():
    return {"status": "running"}
