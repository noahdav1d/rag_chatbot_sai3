from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# Load embedding model
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Load FAISS database
db = FAISS.load_local("faiss_index", embedding_model, allow_dangerous_deserialization=True)

# Create retriever from FAISS index
retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 3})
