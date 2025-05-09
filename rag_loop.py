import gradio as gr
from langchain.chat_models import init_chat_model
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA

# Modell und Datenbank laden
llm = init_chat_model("meta-llama/Llama-3.3-70B-Instruct-Turbo-Free", model_provider="together")
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.load_local("faiss_index", embedding_model, allow_dangerous_deserialization=True)
retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# RAG-Chain erstellen
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

# Funktion, die die Antwort basierend auf der Frage berechnet
def answer_question(question):
    return qa_chain.run(question)

# Gradio UI erstellen
iface = gr.Interface(fn=answer_question, inputs="text", outputs="text")

# Anwendung starten
iface.launch()
