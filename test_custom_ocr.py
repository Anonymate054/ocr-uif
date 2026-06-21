import os
import time
import numpy as np
import cv2
import fitz
import easyocr
import pymupdf4llm

print("Libraries imported. Initializing EasyOCR Reader (Spanish & English)...")
# Initialize the EasyOCR reader. The first run will download model files to ~/.EasyOCR/model/
# once downloaded, they are fully cached locally and run 100% offline.
reader = easyocr.Reader(['es', 'en'], gpu=False) # Force CPU for local sandbox stability
print("EasyOCR Reader initialized.")

def easyocr_plugin(page, pixmap=None, dpi=150, language=None, **kwargs):
    print(f"Running EasyOCR on page {page.number + 1}...")
    
    # If no pixmap is provided, generate one
    if pixmap is None:
        pixmap = page.get_pixmap(dpi=dpi)
        
    # Convert pixmap to numpy array
    img_data = np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(pixmap.h, pixmap.w, pixmap.n)
    
    # Ensure 3 channels RGB
    if pixmap.n == 4:
        img_data = cv2.cvtColor(img_data, cv2.COLOR_RGBA2RGB)
    elif pixmap.n == 1:
        # Grayscale to RGB
        img_data = cv2.cvtColor(img_data, cv2.COLOR_GRAY2RGB)
        
    # Run EasyOCR
    results = reader.readtext(img_data)
    
    # Scale factor from pixmap pixels to PDF points (1 inch = 72 PDF points)
    scale = 72.0 / dpi
    
    # Insert text into the page
    for bbox, text, prob in results:
        # bbox shape: [[x0, y0], [x1, y1], [x2, y2], [x3, y3]]
        x0, y0 = bbox[0]
        x2, y2 = bbox[2]
        
        # Scale back to PDF coordinates
        rect = fitz.Rect(x0 * scale, y0 * scale, x2 * scale, y2 * scale)
        
        # Insert text into the page so pymupdf4llm can read it
        try:
            # We insert the text at the top-left of the bounding box
            # Use a tiny font or transparent text if we were editing, but for extraction standard is fine
            page.insert_text(rect.tl, text, fontsize=9)
        except Exception as e:
            # Catch any positioning or encoding errors
            pass

def main():
    pdf_path = "files/lpb_oficios_01_06_26/8_operadora_y_desarrolladora_de_industrias.pdf"
    print(f"Opening PDF: {pdf_path}")
    
    if not os.path.exists(pdf_path):
        print(f"Error: {pdf_path} not found!")
        return
        
    # 1. Plain Text Extraction using EasyOCR directly (MVP 1)
    print("\n--- Extracting Plain Text (MVP 1) ---")
    start_time = time.time()
    
    doc = fitz.open(pdf_path)
    plain_text_pages = []
    
    for page_num, page in enumerate(doc):
        print(f"Processing Page {page_num + 1}/{len(doc)} for plain text...")
        pix = page.get_pixmap(dpi=150)
        img_data = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
        if pix.n == 4:
            img_data = cv2.cvtColor(img_data, cv2.COLOR_RGBA2RGB)
        results = reader.readtext(img_data)
        page_text = "\n".join([item[1] for item in results])
        plain_text_pages.append(f"--- Page {page_num + 1} ---\n{page_text}")
        
    plain_text = "\n\n".join(plain_text_pages)
    plain_time = time.time() - start_time
    print(f"Plain text extraction completed in {plain_time:.2f} seconds.")
    print(f"Total character count: {len(plain_text)}")
    
    # Save Plain Text
    plain_output_path = "8_operadora_y_desarrolladora_de_industrias_plain.txt"
    with open(plain_output_path, "w", encoding="utf-8") as f:
        f.write(plain_text)
    print(f"Saved MVP 1 (Plain Text) to '{plain_output_path}'")
    
    # 2. Markdown Extraction using PyMuPDF4LLM with custom EasyOCR plugin (MVP 2)
    print("\n--- Extracting Markdown (MVP 2) ---")
    start_time = time.time()
    
    # We pass the custom plugin and force OCR on all pages
    md_text = pymupdf4llm.to_markdown(
        pdf_path,
        ocr_function=easyocr_plugin,
        force_ocr=True,
        dpi=150
    )
    
    md_time = time.time() - start_time
    print(f"Markdown extraction completed in {md_time:.2f} seconds.")
    print(f"Total markdown character count: {len(md_text)}")
    
    # Save Markdown
    markdown_output_path = "8_operadora_y_desarrolladora_de_industrias_markdown.md"
    with open(markdown_output_path, "w", encoding="utf-8") as f:
        f.write(md_text)
    print(f"Saved MVP 2 (Markdown) to '{markdown_output_path}'")
    
    print("\nSnippets of result:")
    print("PLAIN TEXT SNIPPET:")
    print(plain_text[:400])
    print("\nMARKDOWN TEXT SNIPPET:")
    print(md_text[:400])

if __name__ == "__main__":
    main()
