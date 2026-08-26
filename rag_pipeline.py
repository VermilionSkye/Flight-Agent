import os
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()

DATA_DIR = "./data"
PERSIST_DIR = "./vector_db"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

def ingest_documents():
    if not os.path.exists(DATA_DIR) or not os.listdir(DATA_DIR):
        print(f"[!] No documents found in '{DATA_DIR}'. Please drop at least one PDF there.")
        return

    print(f"[*] Loading PDFs from '{DATA_DIR}'...")
    loader = PyPDFDirectoryLoader(DATA_DIR)
    docs = loader.load()
    print(f"[+] Loaded {len(docs)} document pages.")

    # Split documents into semantic chunks
    print("[*] Splitting text into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        length_function=len,
        add_start_index=True
    )
    chunks = text_splitter.split_documents(docs)
    print(f"[+] Created {len(chunks)} chunks.")

    # Initialize local open-source embeddings (Free, CPU-friendly)
    print(f"[*] Initializing embedding model ({EMBEDDING_MODEL_NAME})...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

    # Store and persist chunks in ChromaDB
    print(f"[*] Writing embeddings to ChromaDB at '{PERSIST_DIR}'...")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=PERSIST_DIR,
        collection_name="airline_policies"
    )
    print("[+] Ingestion complete! Vector database is ready.")

if __name__ == "__main__":
    ingest_documents()