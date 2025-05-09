import fitz  # PyMuPDF
import os

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    all_text = ""

    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text = page.get_text()
        all_text += f"\n--- Page {page_num + 1} ---\n{text}"

    doc.close()
    return all_text

# Beispielnutzung
if __name__ == "__main__":
    # Pfad zur PDF-Datei
    base_dir = "C:\\Users\\noahd\\Desktop\\chatbot\\chatbot_sai\\test_data"
    filename = "User_manual_ASSA_ABLOY_RP400_de-DE.pdf"  # <-- ändere das zu deinem echten Dateinamen
    pdf_file = os.path.join(base_dir, filename)

    text = extract_text_from_pdf(pdf_file)

    # Ersten Teil anzeigen
    print(text[:1000])  # Nur ersten Teil anzeigen, damit's nicht zu viel wird
