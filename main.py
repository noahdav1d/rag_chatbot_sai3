from fastapi import FastAPI, Query
from rag_chain import run_rag_chain  # <-- das gibt's wirklich!

app = FastAPI()

@app.get("/")
def root():
    return {"message": "RAG Chatbot API is running"}

@app.get("/ask")
def ask(question: str = Query(..., description="Frage an den RAG-Chatbot")):
    answer = run_rag_chain(question)
    return {"question": question, "answer": answer}
