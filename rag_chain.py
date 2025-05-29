import os
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chat_models import init_chat_model

# .env-Datei laden
load_dotenv()

# API-Key holen
api_key = os.getenv("TOGETHER_API_KEY")
if not api_key:
    raise ValueError("TOGETHER_API_KEY nicht gefunden. Bitte in der .env-Datei setzen.")

os.environ["TOGETHER_API_KEY"] = api_key

# Modell & Retriever initialisieren (nur einmal beim Start)
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
llm = init_chat_model("meta-llama/Llama-3.3-70B-Instruct-Turbo-Free", model_provider="together")
db = FAISS.load_local("faiss_index", embedding_model, allow_dangerous_deserialization=True)
retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# RAG-Chain einmalig bauen
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

# Funktion für externe Nutzung
def run_rag_chain(question: str) -> str:
    return qa_chain.run(question)
