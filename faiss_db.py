from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
import os

# Optional: PDF-Funktionen laden
from extract_pdf import extract_text_from_pdf
from split_text import split_text_into_chunks

# 1. Modell laden
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# 2. PDF laden + Chunks erzeugen
pdf_path = "test_data/User_manual_ASSA_ABLOY_RP400_de-DE.pdf"
text = extract_text_from_pdf(pdf_path)
chunks = split_text_into_chunks(text)

# 3. Chunks in LangChain Document-Objekte umwandeln
documents = [Document(page_content=chunk) for chunk in chunks]

# 4. FAISS Index erstellen
db = FAISS.from_documents(documents, embedding_model)

# 5. Speichern
db.save_local("faiss_index")
print("✅ FAISS-Index gespeichert in 'faiss_index/'")