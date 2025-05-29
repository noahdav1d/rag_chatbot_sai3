from sentence_transformers import SentenceTransformer
import os

# Wähle ein kostenloses Hugging Face Embedding-Modell
model_name = 'sentence-transformers/all-MiniLM-L6-v2'

# Lade das Modell
model = SentenceTransformer(model_name)

def generate_embeddings(chunks):
    # Embeddings für jeden Chunk erzeugen
    embeddings = model.encode(chunks, convert_to_tensor=True)
    return embeddings

# Beispielnutzung
if __name__ == "__main__":
    from split_text import split_text_into_chunks
    from extract_pdf import clean_and_extract_text_from_pdf

    # PDF-Datei einlesen und in Chunks aufteilen
    pdf_path = "./test_data/User_manual_ASSA_ABLOY_RP400_de-DE.pdf"
    raw_text = clean_and_extract_text_from_pdf(pdf_path)
    chunks = split_text_into_chunks(raw_text)

    # Embeddings für Chunks erzeugen
    embeddings = generate_embeddings(chunks)

    print(f"Anzahl der Embeddings: {len(embeddings)}")
    print(f"Beispiel-Embedding (erster Chunk): {embeddings[0][:5]}...")  # Nur ein Teilausdruck des Embeddings