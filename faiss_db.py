from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
import os
from pathlib import Path

# Optional: PDF-Funktionen laden
from extract_pdf import clean_and_extract_text_from_pdf
from split_text import split_text_into_chunks

# 1. Modell laden
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# 2. PDFs laden + Chunks erzeugen
documents = []
pdf_dir = Path("test_data")
for pdf_file in pdf_dir.rglob("*.pdf"):
    print(f"Processing: {pdf_file.name}")
    text = clean_and_extract_text_from_pdf(pdf_file)
    chunks = split_text_into_chunks(text, source_file=pdf_file.name)
    documents.extend([Document(page_content=chunk) for chunk in chunks])

# 4. FAISS Index erstellen
db = FAISS.from_documents(documents, embedding_model)

# 5. Speichern
db.save_local("faiss_index")
print("✅ FAISS-Index gespeichert in 'faiss_index/'")