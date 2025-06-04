import os
import streamlit as st
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA

# Load .env file
load_dotenv()

# Set your Together API key - first .env, then sidebar
api_key = os.getenv("TOGETHER_API_KEY")
if not api_key:
    api_key = st.sidebar.text_input(
        label = "#### Set your Together AI API key here 👇",
        placeholder = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
        type = "password")

# App title
st.title("RAG Chatbot - Assa Abloy")

# Load models only once (Caching)
@st.cache_resource
def load_rag_models():
    if not api_key:
        return None
    
    # Set API key
    os.environ["TOGETHER_API_KEY"] = api_key
    
    # Load model and database
    llm = init_chat_model("meta-llama/Llama-3.3-70B-Instruct-Turbo-Free", model_provider="together")
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.load_local("faiss_index", embedding_model, allow_dangerous_deserialization=True)
    retriever = db.as_retriever(
        search_type="similarity", 
        search_kwargs={"k": 5}  # Increased to 5 for more sources
    )
    
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

    # RAG-Chain without return_source_documents for simpler answers
    qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    chain_type_kwargs={"prompt": prompt_template}
    )
    return qa_chain

# Initialize chat session management
if "chats" not in st.session_state:
    st.session_state.chats = {"Chat 1": []}
if "current_chat" not in st.session_state:
    st.session_state.current_chat = "Chat 1"
if "chat_counter" not in st.session_state:
    st.session_state.chat_counter = 1

# Sidebar for chat management
with st.sidebar:
    st.markdown("### 💬 Chat Management")
    
    # New chat button
    if st.button("➕ New Chat", use_container_width=True):
        st.session_state.chat_counter += 1
        new_chat_name = f"Chat {st.session_state.chat_counter}"
        st.session_state.chats[new_chat_name] = []
        st.session_state.current_chat = new_chat_name
        st.rerun()
    
    # Chat selection
    if len(st.session_state.chats) > 1:
        st.markdown("**Current Chats:**")
        for chat_name in st.session_state.chats.keys():
            col1, col2 = st.columns([3, 1])
            with col1:
                if st.button(chat_name, use_container_width=True, 
                           type="primary" if chat_name == st.session_state.current_chat else "secondary"):
                    st.session_state.current_chat = chat_name
                    st.rerun()
            with col2:
                if st.button("🗑️", key=f"delete_{chat_name}"):
                    if len(st.session_state.chats) > 1:  # Keep at least one chat
                        del st.session_state.chats[chat_name]
                        if st.session_state.current_chat == chat_name:
                            st.session_state.current_chat = list(st.session_state.chats.keys())[0]
                        st.rerun()
    
    st.markdown("---")
    st.markdown("### 🛠️ Chat Options")
    
    if st.button("🗑️ Clear Current Chat", use_container_width=True):
        st.session_state.chats[st.session_state.current_chat] = []
        st.rerun()
    
    st.markdown("---")
    current_messages = st.session_state.chats.get(st.session_state.current_chat, [])
    st.markdown(f"**📊 Messages in {st.session_state.current_chat}:** {len(current_messages)}")
    
    if api_key:
        st.success("✅ API Key set")
    else:
        st.warning("⚠️ API Key missing")

# Display current chat
st.markdown(f"**Current Chat:** {st.session_state.current_chat}")

# Display messages from current chat
current_messages = st.session_state.chats.get(st.session_state.current_chat, [])
for message in current_messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Add user input and response to the current chat
if prompt := st.chat_input("Ask your question here..."):
    # Add user message to current chat
    st.session_state.chats[st.session_state.current_chat].append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    if api_key:
        qa_chain = load_rag_models()
        
        if qa_chain:
            try:
                # Generate RAG response (without source citations)
                with st.spinner("🤔 Thinking..."):
                    response = qa_chain.invoke(prompt)
                
                # Add response to current chat
                st.session_state.chats[st.session_state.current_chat].append({"role": "assistant", "content": response})
                
                # Reload page to show new message
                st.rerun()

            except Exception as e:
                error_message = f"Sorry, an error occurred: {str(e)}"
                st.session_state.chats[st.session_state.current_chat].append({"role": "assistant", "content": error_message})
                st.rerun()
        else:
            error_message = "Error loading models. Please check your API key."
            st.session_state.chats[st.session_state.current_chat].append({"role": "assistant", "content": error_message})
            st.rerun()

    else:
        error_message = "Please enter your Together AI API key in the sidebar."
        st.session_state.chats[st.session_state.current_chat].append({"role": "assistant", "content": error_message})
        st.rerun()