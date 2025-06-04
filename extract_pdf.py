import fitz  # PyMuPDF
import os
from pathlib import Path

def clean_and_extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    toc = doc.get_toc()
    urheberrecht_page = None
    index_page = None

    # Find target pages (convert to 0-based)
    for entry in toc:
        title = entry[1]
        page = entry[2] - 1
        if title == 'Urheberrecht und Haftungsausschluss':
            urheberrecht_page = page
        elif title == 'Index':
            index_page = page

    last_page_idx = doc.page_count - 1

    # Delete Index pages first
    if index_page is not None and index_page <= last_page_idx:
        doc.delete_pages(from_page=index_page, to_page=last_page_idx)
        last_page_idx = doc.page_count - 1

    # Delete Urheberrecht page
    if urheberrecht_page is not None and urheberrecht_page <= last_page_idx:
        doc.delete_page(urheberrecht_page)

    # Extract text
    all_text = ""
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text = page.get_text()
        all_text += f"\n--- Page {page_num + 1} ---\n{text}"

    doc.close()
    return all_text

def process_all_pdfs_in_folder(input_dir, output_dir):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    for pdf_file in input_dir.rglob("*.pdf"):
        print(f"Processing: {pdf_file.name}")
        text = clean_and_extract_text_from_pdf(pdf_file)
        output_text_path = output_dir / (pdf_file.stem + ".txt")
        with open(output_text_path, "w", encoding="utf-8") as out:
            out.write(text)
        print(f"Saved cleaned text to: {output_text_path}")

# Beispielnutzung
if __name__ == "__main__":
    # Passe die Pfade ggf. an
    process_all_pdfs_in_folder("./test_data", "./cleaned_text")