import os
from extract_pdf import clean_and_extract_text_from_pdf

def split_text_into_chunks(text, chunk_size=500, overlap=50):
    """
    Split text into overlapping chunks.

    Args:
        text (str): Full text to split.
        chunk_size (int): Size of each chunk.
        overlap (int): Number of characters to overlap between chunks.

    Returns:
        list of str: List of text chunks.
    """
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

# Optional entry point for batch processing
if __name__ == "__main__":
    folder_path = "./test_data"
    output_folder = "./cleaned_text"
    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(folder_path, filename)
            try:
                raw_text = clean_and_extract_text_from_pdf(pdf_path)
                output_file = os.path.join(output_folder, filename.replace(".pdf", ".txt"))
                with open(output_file, "w", encoding="utf-8") as f:
                    f.write(raw_text)
                print(f"✅ Saved cleaned text to: {output_file}")
            except Exception as e:
                print(f"❌ Failed to process {filename}: {e}")
