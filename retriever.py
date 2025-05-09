from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# 1. Embedding-Modell (muss identisch zum Indexieren sein)
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# 2. Lade die bestehende FAISS-Datenbank (Verzeichnisname beim Speichern angeben!)
db = FAISS.load_local("faiss_index", embedding_model, allow_dangerous_deserialization=True)

# 3. Retriever erstellen
retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# 4. Nutzerfrage
frage = "Wie öffnet man das Tor bei Stromausfall?"

# 5. Relevante Dokument-Chunks abrufen
relevante_docs = retriever.get_relevant_documents(frage)

# 6. Ausgabe
print("🔍 Relevante Chunks:")
for i, doc in enumerate(relevante_docs):
    print(f"\n--- Chunk {i+1} ---")
    print(doc.page_content)