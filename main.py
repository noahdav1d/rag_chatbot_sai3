from fastapi import FastAPI, Query
from rag_chain import run_rag_chain

app = FastAPI()

@app.get("/")
def root():
    return {"message": "RAG Chatbot API is running"}

@app.get("/ask")
def ask(question: str = Query(..., description="Ask a question to the RAG chatbot")):
    # Calls the RAG chain with the user's question and returns the answer
    answer = run_rag_chain(question)
    return {"question": question, "answer": answer}
