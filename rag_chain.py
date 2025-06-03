import os
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chat_models import init_chat_model
from retriever import retriever

# Load .env file
load_dotenv()

# Get API key from .env file
api_key = os.getenv("TOGETHER_API_KEY")
if not api_key:
    raise ValueError("TOGETHER_API_KEY not found. Please set it in the .env file.")

# Provide key for model access
os.environ["TOGETHER_API_KEY"] = api_key

# 1. Load model
llm = init_chat_model("meta-llama/Llama-3-70B-Instruct", model_provider="together")

# 2. Create RAG chain
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

# 3. Function to call the chain with a question
def run_rag_chain(question: str) -> str:
    # Returns the answer from the RAG chain
    return qa_chain.run(question)
