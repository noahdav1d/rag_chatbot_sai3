#!/bin/bash
python extract_pdf.py
python split_text.py
python faiss_db.py
python retriever.py
echo "✅ FAISS index is ready."


