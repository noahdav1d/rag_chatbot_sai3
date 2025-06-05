FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python packages
COPY requirements-lock.txt requirements.txt
RUN pip install -r requirements.txt

#RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy all project files including setup.sh
COPY . .

# Ensure setup.sh is executable
RUN chmod +x run_setup.sh

# Run setup and then start the Streamlit chatbot
CMD ["bash", "-c", "./run_setup.sh && streamlit run rag_loop.py --server.port 7860 --server.address 0.0.0.0"]
