# Basisimage mit Python
FROM python:3.10-slim

# Arbeitsverzeichnis
WORKDIR /app

# Systemabhängigkeiten
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Projektdateien kopieren
COPY . /app

# Python-Abhängigkeiten installieren
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# .env-Variablen laden (für Together API)
ENV PYTHONUNBUFFERED=1

# Port für Gradio
EXPOSE 7860

# Startkommando
CMD ["python", "rag_loop.py"]
