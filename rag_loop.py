import os
import gradio as gr
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chat_models import init_chat_model

# Load .env file to get API key
load_dotenv()

# Get Together API key from environment
api_key = os.getenv("TOGETHER_API_KEY")
if not api_key:
    raise ValueError("TOGETHER_API_KEY not found. Please set it in the .env file.")

# Set the environment variable so Together API can use it
os.environ["TOGETHER_API_KEY"] = api_key

# 1. Load the language model via Together API
llm = init_chat_model("meta-llama/Llama-3-70B-Instruct", model_provider="together")

# 2. Load the FAISS database with precomputed embeddings
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.load_local("faiss_index", embedding_model, allow_dangerous_deserialization=True)
retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# 3. Build the RAG chain (retriever + LLM)
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

# 4. Define the chatbot response function
def chatbot(question):
    # Get the answer from the RAG chain
    answer = qa_chain.run(question)
    return answer

# 5. Build the Gradio interface
iface = gr.Interface(
    fn=chatbot,
    inputs=gr.Textbox(lines=2, placeholder="Ask your question here..."),
    outputs="text",
    title="RAG Chatbot for ASSA ABLOY",
    description="Ask a question and receive an answer based on ASSA ABLOY documentation."
)

# 6. Launch the Gradio app (make accessible on all interfaces, port 7860)
iface.launch(server_name="0.0.0.0", server_port=7860)
