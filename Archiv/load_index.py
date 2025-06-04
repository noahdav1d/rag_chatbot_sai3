from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Sicheres Laden (nur wenn du der Quelle vertraust!)
db = FAISS.load_local("faiss_index", embedding_model, allow_dangerous_deserialization=True)

# Test: Ähnlichkeitssuche
query = "Wie öffne ich das Tor manuell?"
results = db.similarity_search(query, k=3)

for i, result in enumerate(results, 1):
    print(f"\n🔎 Treffer {i}:\n{result.page_content}")
