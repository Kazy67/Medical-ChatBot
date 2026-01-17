from src.helper import load_pdf, text_split, download_huggingface_embeddings
from langchain_chroma import Chroma
import os

# This script is for INITIAL setup
extracted_data = load_pdf("data/")
text_chunks = text_split(extracted_data)
embeddings = download_huggingface_embeddings()

# Create and PERSIST the database to the 'db' folder
docsearch = Chroma.from_documents(
    documents=text_chunks,
    embedding=embeddings,
    persist_directory='db'
)
print("Database created and saved locally!")