import os
import sys

def check_pdf(pdf_path):
    print(f"Checking PDF file: {pdf_path}")
    if not os.path.exists(pdf_path):
        print(f"Error: File not found at {pdf_path}")
        return
    
    # Try importing pypdf first (pure Python, usually safe to import even during installations)
    try:
        import pypdf
        reader = pypdf.PdfReader(pdf_path)
        num_pages = len(reader.pages)
        print(f"Total pages (pypdf): {num_pages}")
        
        total_text = ""
        for i, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            total_text += text
            print(f"  Page {i+1}: extracted {len(text)} characters")
            
        print(f"Total extracted characters: {len(total_text)}")
        if len(total_text.strip()) > 50:
            print("Verdict: This PDF contains a selectable text layer! We can extract it 100% locally and accurately.")
        else:
            print("Verdict: Scanned PDF or empty file. OCR will be required.")
            
    except ImportError:
        print("pypdf is not available yet in the Python path.")
        
    # Also try PyMuPDF if available
    try:
        import fitz
        doc = fitz.open(pdf_path)
        print(f"\nTotal pages (PyMuPDF): {len(doc)}")
        for page_num, page in enumerate(doc):
            text = page.get_text()
            print(f"  Page {page_num+1}: extracted {len(text)} characters")
    except ImportError:
        pass

if __name__ == "__main__":
    pdf_path = "/home/lenovo/Documents/projects/ocr-uif/files/lpb_oficios_01_06_26/8_operadora_y_desarrolladora_de_industrias.pdf"
    check_pdf(pdf_path)
