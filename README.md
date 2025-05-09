<h1>Chatbot with RAG and LangChain</h1>

<h2>Prerequisites</h2>
<ul>
  <li>Python 3.11+</li>
</ul>

<h2>Installation</h2>
<h3>1. Clone the repository:</h3>

```
git clone https://github.com/noahdav1d/rag_chatbot_sai3.git
cd rag_chatbot_sai3
```

<h3>2. Create a virtual environment</h3>

```
python -m venv venv
```

<h3>3. Activate the virtual environment</h3>

```
venv\Scripts\Activate
(or on Mac): source venv/bin/activate
```

<h3>4. Install libraries</h3>

```
pip install -r requirements.txt
```

<h3>5. Add Together API Key</h3>
Rename the .env file to .env
Add your Together API Key

<h2>Executing the scripts</h2>

- Open a terminal in VS Code

- Execute the following command:

```
python extract_pdf.py
python split_text.py
python embeddings.py
python faiss_db.py
python load_index.py
python retriever.py
python rag_chain.py
python rag_loop.py
```
