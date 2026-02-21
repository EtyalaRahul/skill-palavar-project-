import os
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

DB_PATH = "vector_db"

URLS = [
    "https://stackoverflow.com/questions/43170942/laptop-overheating-on-startup"
]

print("🔗 Loading web pages...")

loader = WebBaseLoader(URLS)
documents = loader.load()

print(f" Loaded {len(documents)} web documents")


splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=100
)

chunks = splitter.split_documents(documents)
print(f"✂️ Split into {len(chunks)} chunks")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

db = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory=DB_PATH
)

db.persist()
print("✅ Web data ingested into vector database")