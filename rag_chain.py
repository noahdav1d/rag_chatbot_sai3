import getpass
import os
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chat_models import init_chat_model

# API-Key setzen, falls noch nicht vorhanden
if not os.environ.get("TOGETHER_API_KEY"):
    os.environ["TOGETHER_API_KEY"] = getpass.getpass("Enter API key for Together AI: ")

# 1. Modell laden
llm = init_chat_model("meta-llama/Llama-3.3-70B-Instruct-Turbo-Free", model_provider="together")

# 2. FAISS-Datenbank laden
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.load_local("faiss_index", embedding_model, allow_dangerous_deserialization=True)
retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# 3. RAG-Chain erstellen
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

# 4. Testfrage stellen
frage = "Was sind allgemeine Sicherheitshinweise?"
antwort = qa_chain.run(frage)

print("🤖 Antwort:")
print(antwort)