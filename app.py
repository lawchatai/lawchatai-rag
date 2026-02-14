import os
import re
import hashlib
import tempfile
import shutil

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

import boto3
from botocore.client import Config

from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    Docx2txtLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import HuggingFaceEmbeddings


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
# ENV (SAFE)
# =========================

required_env = [
    "R2_ACCOUNT_ID",
    "R2_ACCESS_KEY_ID",
    "R2_SECRET_ACCESS_KEY",
    "R2_BUCKET_NAME",
    "GEMINI_API_KEY",
]

for var in required_env:
    if not os.getenv(var):
        print(f"WARNING: Missing environment variable: {var}")

# =========================
# GLOBALS (INITIALIZED ON STARTUP)
# =========================

llm = None
embeddings = None
s3 = None

# =========================
# STARTUP EVENT
# =========================

@app.on_event("startup")
async def startup_event():
    global llm, embeddings, s3

    print("Starting up application...")

    # Init S3 (R2)
    try:
        s3 = boto3.client(
            "s3",
            endpoint_url=f"https://{os.getenv('R2_ACCOUNT_ID')}.r2.cloudflarestorage.com",
            aws_access_key_id=os.getenv("R2_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("R2_SECRET_ACCESS_KEY"),
            config=Config(signature_version="s3v4"),
        )
        print("R2 connected")
    except Exception as e:
        print("R2 init failed:", str(e))

    # Init LLM
    try:
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0.1,
        )
        print("LLM initialized")
    except Exception as e:
        print("LLM init failed:", str(e))

    # Init Embeddings (may take time first deploy)
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
        )
        print("Embeddings initialized")
    except Exception as e:
        print("Embeddings init failed:", str(e))


# =========================
# HELPERS
# =========================

ALLOWED_EXTENSIONS = {".pdf", ".txt", ".docx"}

def validate_user_id(user_id: str):
    if not re.match(r"^[a-zA-Z0-9_-]{3,50}$", user_id):
        raise ValueError("Invalid user_id")

def get_file_hash(file_bytes: bytes):
    return hashlib.sha256(file_bytes).hexdigest()

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
# HEALTH (FAST)
# =========================

@app.get("/")
def health():
    return {"status": "running"}

# =========================
# UPLOAD
# =========================

@app.post("/upload")
async def upload_document(
    user_id: str = Form(...),
    file: UploadFile = File(...),
):
    if not s3 or not embeddings:
        return JSONResponse({"error": "Server not ready"}, status_code=503)

    try:
        validate_user_id(user_id)
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=400)

    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        return JSONResponse({"error": "Unsupported file type"}, status_code=400)

    try:
        file_bytes = await file.read()
        file_hash = get_file_hash(file_bytes)

        file_key = f"users/{user_id}/files/{file_hash}_{file.filename}"
        index_prefix = f"users/{user_id}/index/"

        s3.put_object(Bucket=os.getenv("R2_BUCKET_NAME"), Key=file_key, Body=file_bytes)

        temp_dir = tempfile.gettempdir()
        temp_path = os.path.join(temp_dir, f"{file_hash}_{file.filename}")

        with open(temp_path, "wb") as f:
            f.write(file_bytes)

        docs = load_document(temp_path, ext)

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )

        documents = splitter.split_documents(docs)

        vectorstore = FAISS.from_documents(documents, embeddings)

        local_index_dir = os.path.join(temp_dir, f"faiss_{user_id}")
        os.makedirs(local_index_dir, exist_ok=True)

        vectorstore.save_local(local_index_dir)

        for fname in os.listdir(local_index_dir):
            s3.upload_file(
                os.path.join(local_index_dir, fname),
                os.getenv("R2_BUCKET_NAME"),
                f"{index_prefix}{fname}",
            )

        os.remove(temp_path)

        return {"status": "success", "message": "Uploaded & indexed"}

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


# =========================
# QUERY
# =========================

@app.post("/query")
async def query_documents(
    user_id: str = Form(...),
    question: str = Form(...),
):
    if not s3 or not embeddings or not llm:
        return JSONResponse({"error": "Server not ready"}, status_code=503)

    try:
        validate_user_id(user_id)

        temp_dir = tempfile.gettempdir()
        local_index_dir = os.path.join(temp_dir, f"faiss_{user_id}")
        os.makedirs(local_index_dir, exist_ok=True)

        objs = s3.list_objects_v2(
            Bucket=os.getenv("R2_BUCKET_NAME"),
            Prefix=f"users/{user_id}/index/",
        )

        if "Contents" not in objs:
            return JSONResponse({"error": "No documents indexed"}, status_code=400)

        for obj in objs["Contents"]:
            key = obj["Key"]
            local_file = os.path.join(local_index_dir, key.split("/")[-1])
            s3.download_file(os.getenv("R2_BUCKET_NAME"), key, local_file)

        vectorstore = FAISS.load_local(
            local_index_dir,
            embeddings,
            allow_dangerous_deserialization=True
        )

        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        docs = retriever.get_relevant_documents(question)

        if not docs:
            return {"answer": "No relevant information found in the document."}

        context = "\n\n".join([doc.page_content for doc in docs])

        prompt = f"""
        You are a legal document analysis assistant.

        STRICT RULES:
        - Answer ONLY using the provided context
        - If the answer is not explicitly stated, respond exactly with:
          "The document does not contain this information."

        Context:
        {context}

        Question:
        {question}

        Answer:
        """

        response = llm.invoke(prompt)

        return {"answer": response.content}

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
