<h1>Chatbot with RAG and LangChain</h1>

<h2>Prerequisites</h2>
<ul>
  <li>Python 3.11+ and Bash</li>
  <li> or </li>
  <li>Docker</li>
  <li>Git</li>
</ul>

<h2>Installation</h2>
<h3>1. Clone the repository:</h3>

```
git clone https://github.com/noahdav1d/rag_chatbot_sai3.git
cd rag_chatbot_sai3
```
<h2>Python setup</h2>
<h3>2. Create a virtual environment</h3>

```
python -m venv venv
(or on Mac): python3 -m venv venv
```

<h3>3. Activate the virtual environment</h3>

```
venv\Scripts\Activate
(or on Mac): source venv/bin/activate
```

<h3>4. Install libraries</h3>

- Open a terminal in VS Code

```
pip install -r requirements.txt
```

<h2>Executing the script</h2>

- Execute the following command:

```
./run_setup.sh
```

<h2>Docker Setup</h2>
```
docker-compose up --build
```