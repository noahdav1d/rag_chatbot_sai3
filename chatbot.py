import getpass
import os

# API-Key abfragen, falls nicht gesetzt
if not os.environ.get("TOGETHER_API_KEY"):
    os.environ["TOGETHER_API_KEY"] = getpass.getpass("Enter API key for Together AI: ")

from langchain.chat_models import init_chat_model

# Modell laden
model = init_chat_model("meta-llama/Llama-3.3-70B-Instruct-Turbo-Free", model_provider="together")

# Modellanfrage senden und Antwort speichern
response = model.invoke("Hello, world!")

# Antwort anzeigen
print(response)