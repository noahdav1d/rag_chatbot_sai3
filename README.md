<h1>Chatbot with RAG and LangChain</h1>

<h2>Prerequisites</h2>
<ul>
  <li>Git</li>
  <li>Python 3.11+ and Bash</li>
  <li> or </li>
  <li>Docker</li>
</ul>

<h2> Docker And Python</h2>
<h2>Installation</h2>
<h3>1. Clone the repository:</h3>

```
git clone https://github.com/noahdav1d/rag_chatbot_sai3.git
cd rag_chatbot_sai3
```

<h2>Togeteher API Key</h2>
<h3>2. Add Together API Key</h3>

```
Rename the .env.example file to .env and Add your Together API Key
```

<h2>Python setup</h2>
<h3>3. Create a virtual environment</h3>

```
python -m venv venv
(or on Mac): python3 -m venv venv
```

<h3>4. Activate the virtual environment</h3>

```
venv\Scripts\Activate
(or on Mac): source venv/bin/activate
```

<h3>5. Install libraries</h3>

- Open a terminal in VS Code

```
pip install -r requirements.txt
```

<h3>6. Executing the script</h2>

- Execute the following command:

```
./run_setup_python.sh
```

<h2>Docker Setup</h2>
<h3>3. Start Docker Compose</h3>

```
docker-compose up --build
```