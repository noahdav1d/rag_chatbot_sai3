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
retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 5})

# Prompt template
prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are a helpful assistant that helps answering questions about the company "ASSA ABLOY Entrance Systems" and their products.
The Documents provided are user manuals and technical specifications of the products.
The prouducts are installations that you typically find in distribution centers and warehouses, such as sectional doors, high speed doors and docking solutions.
Every product archetype consists of different types that have one or more differences in the technical specifications that make them unique. The product type is mentionend in the file name and should be used as a reference.
Answer the questions like a human expert in the field of door systems and door technology would do.
Answer the question directly wihtout mentioning the source of the information.
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

# Function that creates the answer and returns both answer and sources
def answer_question(question):
    # Get the relevant documents
    docs = retriever.get_relevant_documents(question)
    
    # Get the answer
    answer = qa_chain.run(question)
    
    # Format the sources
    sources = "\n\nSources used:\n"
    for i, doc in enumerate(docs, 1):
        sources += f"\n--- Source {i} ---\n{doc.page_content}\n"
    
    # Combine answer and sources
    full_response = f"Answer:\n{answer}\n{sources}"
    
    return full_response

# Create Gradio Interface with larger output text box
iface = gr.Interface(
    fn=answer_question, 
    inputs="text", 
    outputs=gr.Textbox(lines=10),
    title="ASSA ABLOY Product Assistant",
    description="Ask questions about ASSA ABLOY products and manuals"
)

# Start the Gradio Interface
iface.launch()