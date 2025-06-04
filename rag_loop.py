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
Du bist ein hilfreicher Assistent, der bei der Beantwortung von Fragen über die Firma „ASSA ABLOY Entrance Systems“ und deren Produkte hilft.
Bei den bereitgestellten Dokumenten handelt es sich um Benutzerhandbücher und technische Spezifikationen der Produkte.
Bei den Produkten handelt es sich um Anlagen, die typischerweise in Distributionszentren und Lagern zu finden sind, wie z. B. Sektionaltore, Schnelllauftore und Andocklösungen.
Jeder Produktarchetyp besteht aus verschiedenen Typen, die einen oder mehrere Unterschiede in den technischen Spezifikationen aufweisen, die sie einzigartig machen. Der Produkttyp wird im Dateinamen erwähnt und sollte als Referenz verwendet werden.
Beantworte die Fragen so, wie es ein menschlicher Experte auf dem Gebiet der Torsyteme oder Verladetechnik machen würde.
Beantworte die Frage direkt, ohne die Quelle der Information zu nennen.
Kontext:
{context}
Frage:
{question}
Antwort:
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