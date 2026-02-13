import os
import re
import hashlib
import tempfile
import shutil

from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

import boto3
from botocore.client import Config

# LangChain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    Docx2txtLoader,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA

# =========================
# ENV
# =========================

load_dotenv()

required_env = [
    "R2_ACCOUNT_ID",
    "R2_ACCESS_KEY_ID",
    "R2_SECRET_ACCESS_KEY",
    "R2_BUCKET_NAME",
    "GEMINI_API_KEY",
]

for var in required_env:
    if not os.getenv(var):
        raise ValueError(f"Missing environment variable: {var}")

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
# APP
# =========================

app = FastAPI(title="LawChatAI – LangChain MVP")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# HELPERS
# =========================

def validate_user_id(user_id: str):
    if not re.match(r"^[a-zA-Z0-9_-]{3,50}$", user_id):
        raise ValueError("Invalid user_id")


def get_file_hash(file_bytes: bytes):
    return hashlib.sha256(file_bytes).hexdigest()


def init_models():
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0.1,
    )

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
    )

    return llm, embeddings


def load_document(temp_path: str, ext: str):
    if ext == ".pdf":
        loader = PyPDFLoader(temp_path)
    elif ext == ".txt":
        loader = TextLoader(temp_path)
    elif ext == ".docx":
        loader = Docx2txtLoader(temp_path)
    else:
        raise ValueError("Unsupported file type")

    return loader.load()


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

    file_bytes = await file.read()
    file_hash = get_file_hash(file_bytes)

    file_key = f"users/{user_id}/files/{file_hash}_{file.filename}"
    index_prefix = f"users/{user_id}/index/"

    # Check if already exists
    try:
        s3.head_object(Bucket=R2_BUCKET, Key=file_key)
        return {"status": "success", "message": "File already uploaded"}
    except:
        pass

    # Upload file to R2
    s3.put_object(
        Bucket=R2_BUCKET,
        Key=file_key,
        Body=file_bytes,
    )

    # Save temporarily
    temp_dir = tempfile.gettempdir()
    temp_path = os.path.join(temp_dir, f"{file_hash}_{file.filename}")

    with open(temp_path, "wb") as f:
        f.write(file_bytes)

    # Load + Split
    llm, embeddings = init_models()
    docs = load_document(temp_path, ext)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    documents = splitter.split_documents(docs)

    # Create FAISS
    vectorstore = FAISS.from_documents(documents, embeddings)

    local_index_dir = os.path.join(temp_dir, f"faiss_{user_id}")
    os.makedirs(local_index_dir, exist_ok=True)

    vectorstore.save_local(local_index_dir)

    # Upload FAISS to R2
    for fname in os.listdir(local_index_dir):
        s3.upload_file(
            os.path.join(local_index_dir, fname),
            R2_BUCKET,
            f"{index_prefix}{fname}",
        )

    os.remove(temp_path)

    return {
        "status": "success",
        "message": "Uploaded & indexed",
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
    except ValueError:
        return JSONResponse({"error": "Invalid user_id"}, status_code=400)

    llm, embeddings = init_models()

    temp_dir = tempfile.gettempdir()
    local_index_dir = os.path.join(temp_dir, f"faiss_{user_id}")
    os.makedirs(local_index_dir, exist_ok=True)

    objs = s3.list_objects_v2(
        Bucket=R2_BUCKET,
        Prefix=f"users/{user_id}/index/",
    )

    if "Contents" not in objs:
        return JSONResponse({"error": "No documents indexed"}, status_code=400)

    for obj in objs["Contents"]:
        key = obj["Key"]
        local_file = os.path.join(local_index_dir, key.split("/")[-1])
        s3.download_file(R2_BUCKET, key, local_file)

    vectorstore = FAISS.load_local(
        local_index_dir,
        embeddings,
        allow_dangerous_deserialization=True
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
    )

    result = qa_chain.run(question)

    return {
        "answer": result,
    }


# =========================
# HEALTH
# =========================

@app.get("/")
def health():
    return {"status": "running"}
