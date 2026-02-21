import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

DATA_PATH = "data"
DB_PATH = "vector_db"

documents = []

for file in os.listdir(DATA_PATH):
    path = os.path.join(DATA_PATH, file)

    if file.endswith(".pdf"):
        documents.extend(PyPDFLoader(path).load())
    elif file.endswith(".txt"):
        documents.extend(TextLoader(path).load())

splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=100
)

chunks = splitter.split_documents(documents)

print("Total chunks created:", len(chunks))

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

db = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory=DB_PATH
)

db.persist()
print("Ingestion completed successfully")