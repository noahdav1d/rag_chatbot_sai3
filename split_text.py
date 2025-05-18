from langchain.text_splitter import RecursiveCharacterTextSplitter

def split_text_into_chunks(text, chunk_size=1000, chunk_overlap=200):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )

    chunks = text_splitter.split_text(text)
    return chunks

# Beispielnutzung
if __name__ == "__main__":
    from extract_pdf import clean_and_extract_text_from_pdf

    # PDF-Datei einlesen
    pdf_path = "./test_data\\User_manual_ASSA_ABLOY_RP400_de-DE.pdf"
    raw_text = clean_and_extract_text_from_pdf(pdf_path)

    # Text in Chunks aufteilen
    chunks = split_text_into_chunks(raw_text)

    # Beispielausgabe
    print(f"Anzahl Chunks: {len(chunks)}")
    print("\nErster Chunk:\n")
    print(chunks[0])
