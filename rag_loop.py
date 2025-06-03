import gradio as gr
from langchain.chat_models import init_chat_model
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
load_dotenv()

# Load Model and Database
llm = init_chat_model("meta-llama/Llama-3.3-70B-Instruct-Turbo-Free", model_provider="together")
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.load_local("faiss_index", embedding_model, allow_dangerous_deserialization=True)
retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# Prompt template
prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are a helpful assistant that helps answering questions about the company "ASSA ABLOY" and their products.
The Documents provided are user manuals and technical specifications of the products.
The Products are doors and door systems of different types and sizes. 
Answer the questions like a human expert in the field of door systems and door technology would do.
Please dont mention the documents in your answer, just answer the question directly.
Context:
{context}
Question:
{question}
Answer:
"""
)

# Create Rag-Chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    chain_type_kwargs={"prompt": prompt_template}
)

# Function that creates the answer to a question
def answer_question(question):
    return qa_chain.run(question)
# Create Gradio Interface
iface = gr.Interface(fn=answer_question, inputs="text", outputs="text")
# Start the Gradio Interface
iface.launch()