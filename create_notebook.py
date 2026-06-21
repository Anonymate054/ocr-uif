import json

def create_notebook():
    notebook = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# Local OCR & PDF Text Extraction Sandbox\n",
                    "This notebook evaluates different local libraries for extracting text from PDF files. Since these documents contain sensitive information, all models and processing pipelines run locally.\n",
                    "\n",
                    "## Evaluated Methods:\n",
                    "1. **PyMuPDF (fitz)**: High-speed, robust C-based library to extract digital text layers.\n",
                    "2. **pdfplumber**: Excellent for digital text layers and table layout structure preservation.\n",
                    "3. **pypdf**: Lightweight, pure-python digital text extractor.\n",
                    "4. **pymupdf4llm**: Specialized tool to output layout-aware Markdown format (including tables).\n",
                    "5. **EasyOCR**: Local Deep Learning OCR model (CRAFT detector + CRNN recognizer) to extract text from scanned pages/images."
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 1. Import Libraries"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "import os\n",
                    "import time\n",
                    "import pandas as pd\n",
                    "import numpy as np\n",
                    "import fitz  # PyMuPDF\n",
                    "import pdfplumber\n",
                    "import pypdf\n",
                    "import pymupdf4llm\n",
                    "import easyocr\n",
                    "\n",
                    "print(\"All libraries imported successfully!\")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 2. Configuration & Paths"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "PDF_PATH = \"files/lpb_oficios_01_06_26/8_operadora_y_desarrolladora_de_industrias.pdf\"\n",
                    "print(f\"Target PDF Path: {PDF_PATH}\")\n",
                    "print(f\"File exists: {os.path.exists(PDF_PATH)}\")\n",
                    "if os.path.exists(PDF_PATH):\n",
                    "    print(f\"File size: {os.path.getsize(PDF_PATH) / 1024:.2f} KB\")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 3. Check PDF Properties\n",
                    "Let's check if the PDF has a selectable text layer (digital) or if it is purely an image (scanned)."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "def analyze_pdf_type(pdf_path):\n",
                    "    doc = fitz.open(pdf_path)\n",
                    "    total_chars = 0\n",
                    "    pages_with_text = 0\n",
                    "    for page_num, page in enumerate(doc):\n",
                    "        text = page.get_text()\n",
                    "        char_count = len(text.strip())\n",
                    "        total_chars += char_count\n",
                    "        if char_count > 0:\n",
                    "            pages_with_text += 1\n",
                    "            \n",
                    "    print(f\"Total pages: {len(doc)}\")\n",
                    "    print(f\"Pages with selectable text: {pages_with_text}\")\n",
                    "    print(f\"Total character count: {total_chars}\")\n",
                    "    \n",
                    "    if total_chars > 100:\n",
                    "        print(\"Conclusion: This is a DIGITAL PDF with a selectable text layer.\")\n",
                    "    else:\n",
                    "        print(\"Conclusion: This is a SCANNED PDF (images). OCR is required to extract text.\")\n",
                    "        \n",
                    "analyze_pdf_type(PDF_PATH)"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 4. Define Extraction Functions"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# 1. PyMuPDF Plain Text\n",
                    "def extract_pymupdf(pdf_path):\n",
                    "    doc = fitz.open(pdf_path)\n",
                    "    text = \"\"\n",
                    "    for page_num, page in enumerate(doc):\n",
                    "        text += f\"--- Page {page_num + 1} ---\\n\"\n",
                    "        text += page.get_text() + \"\\n\"\n",
                    "    return text\n",
                    "\n",
                    "# 2. pdfplumber Plain Text\n",
                    "def extract_pdfplumber(pdf_path):\n",
                    "    text = \"\"\n",
                    "    with pdfplumber.open(pdf_path) as pdf:\n",
                    "        for page_num, page in enumerate(pdf.pages):\n",
                    "            text += f\"--- Page {page_num + 1} ---\\n\"\n",
                    "            page_text = page.extract_text()\n",
                    "            if page_text:\n",
                    "                text += page_text + \"\\n\"\n",
                    "    return text\n",
                    "\n",
                    "# 3. pypdf Plain Text\n",
                    "def extract_pypdf(pdf_path):\n",
                    "    reader = pypdf.PdfReader(pdf_path)\n",
                    "    text = \"\"\n",
                    "    for page_num, page in enumerate(reader.pages):\n",
                    "        text += f\"--- Page {page_num + 1} ---\\n\"\n",
                    "        page_text = page.extract_text()\n",
                    "        if page_text:\n",
                    "            text += page_text + \"\\n\"\n",
                    "    return text\n",
                    "\n",
                    "# 4. pymupdf4llm Markdown with EasyOCR plugin\n",
                    "def easyocr_plugin(page, pixmap=None, dpi=150, language=None, **kwargs):\n",
                    "    global _easyocr_reader_global\n",
                    "    if '_easyocr_reader_global' not in globals():\n",
                    "        _easyocr_reader_global = easyocr.Reader(['es', 'en'], gpu=False)\n",
                    "    if pixmap is None:\n",
                    "        pixmap = page.get_pixmap(dpi=dpi)\n",
                    "    img_data = np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(pixmap.h, pixmap.w, pixmap.n)\n",
                    "    if pixmap.n == 4:\n",
                    "        import cv2\n",
                    "        img_data = cv2.cvtColor(img_data, cv2.COLOR_RGBA2RGB)\n",
                    "    elif pixmap.n == 1:\n",
                    "        import cv2\n",
                    "        img_data = cv2.cvtColor(img_data, cv2.COLOR_GRAY2RGB)\n",
                    "    results = _easyocr_reader_global.readtext(img_data)\n",
                    "    scale = 72.0 / dpi\n",
                    "    for bbox, text, prob in results:\n",
                    "        x0, y0 = bbox[0]\n",
                    "        x2, y2 = bbox[2]\n",
                    "        rect = fitz.Rect(x0 * scale, y0 * scale, x2 * scale, y2 * scale)\n",
                    "        try:\n",
                    "            page.insert_text(rect.tl, text, fontsize=9)\n",
                    "        except:\n",
                    "            pass\n",
                    "\n",
                    "def extract_pymupdf4llm(pdf_path):\n",
                    "    doc = fitz.open(pdf_path)\n",
                    "    total_chars = sum([len(page.get_text().strip()) for page in doc])\n",
                    "    if total_chars > 100:\n",
                    "        return pymupdf4llm.to_markdown(pdf_path)\n",
                    "    else:\n",
                    "        return pymupdf4llm.to_markdown(pdf_path, ocr_function=easyocr_plugin, force_ocr=True, dpi=150)\n",
                    "\n",
                    "# 5. EasyOCR (scanned pages fallback)\n",
                    "def extract_easyocr(pdf_path):\n",
                    "    reader = easyocr.Reader(['es', 'en'], gpu=False)\n",
                    "    doc = fitz.open(pdf_path)\n",
                    "    text = \"\"\n",
                    "    for page_num, page in enumerate(doc):\n",
                    "        text += f\"--- Page {page_num + 1} ---\\n\"\n",
                    "        pix = page.get_pixmap(dpi=150)\n",
                    "        img_data = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)\n",
                    "        \n",
                    "        if pix.n == 4:\n",
                    "            import cv2\n",
                    "            img_data = cv2.cvtColor(img_data, cv2.COLOR_RGBA2RGB)\n",
                    "            \n",
                    "        results = reader.readtext(img_data)\n",
                    "        page_text = \"\\n\".join([item[1] for item in results])\n",
                    "        text += page_text + \"\\n\"\n",
                    "    return text"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 5. Evaluate and Compare Models\n",
                    "Let's run each extraction method, measure execution time, and check extraction results."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "results = {}\n",
                    "\n",
                    "# Test PyMuPDF\n",
                    "print(\"Running PyMuPDF...\")\n",
                    "start = time.time()\n",
                    "try:\n",
                    "    text_pymupdf = extract_pymupdf(PDF_PATH)\n",
                    "    results[\"PyMuPDF\"] = {\n",
                    "        \"time\": time.time() - start,\n",
                    "        \"char_count\": len(text_pymupdf),\n",
                    "        \"text\": text_pymupdf,\n",
                    "        \"success\": True,\n",
                    "        \"output_type\": \"Plain Text\"\n",
                    "    }\n",
                    "except Exception as e:\n",
                    "    results[\"PyMuPDF\"] = {\"time\": 0, \"char_count\": 0, \"text\": \"\", \"success\": False, \"error\": str(e), \"output_type\": \"Plain Text\"}\n",
                    "\n",
                    "# Test pdfplumber\n",
                    "print(\"Running pdfplumber...\")\n",
                    "start = time.time()\n",
                    "try:\n",
                    "    text_pdfplumber = extract_pdfplumber(PDF_PATH)\n",
                    "    results[\"pdfplumber\"] = {\n",
                    "        \"time\": time.time() - start,\n",
                    "        \"char_count\": len(text_pdfplumber),\n",
                    "        \"text\": text_pdfplumber,\n",
                    "        \"success\": True,\n",
                    "        \"output_type\": \"Plain Text\"\n",
                    "    }\n",
                    "except Exception as e:\n",
                    "    results[\"pdfplumber\"] = {\"time\": 0, \"char_count\": 0, \"text\": \"\", \"success\": False, \"error\": str(e), \"output_type\": \"Plain Text\"}\n",
                    "\n",
                    "# Test pypdf\n",
                    "print(\"Running pypdf...\")\n",
                    "start = time.time()\n",
                    "try:\n",
                    "    text_pypdf = extract_pypdf(PDF_PATH)\n",
                    "    results[\"pypdf\"] = {\n",
                    "        \"time\": time.time() - start,\n",
                    "        \"char_count\": len(text_pypdf),\n",
                    "        \"text\": text_pypdf,\n",
                    "        \"success\": True,\n",
                    "        \"output_type\": \"Plain Text\"\n",
                    "    }\n",
                    "except Exception as e:\n",
                    "    results[\"pypdf\"] = {\"time\": 0, \"char_count\": 0, \"text\": \"\", \"success\": False, \"error\": str(e), \"output_type\": \"Plain Text\"}\n",
                    "\n",
                    "# Test pymupdf4llm\n",
                    "print(\"Running PyMuPDF4LLM...\")\n",
                    "start = time.time()\n",
                    "try:\n",
                    "    text_pymupdf4llm = extract_pymupdf4llm(PDF_PATH)\n",
                    "    results[\"PyMuPDF4LLM\"] = {\n",
                    "        \"time\": time.time() - start,\n",
                    "        \"char_count\": len(text_pymupdf4llm),\n",
                    "        \"text\": text_pymupdf4llm,\n",
                    "        \"success\": True,\n",
                    "        \"output_type\": \"Markdown\"\n",
                    "    }\n",
                    "except Exception as e:\n",
                    "    results[\"PyMuPDF4LLM\"] = {\"time\": 0, \"char_count\": 0, \"text\": \"\", \"success\": False, \"error\": str(e), \"output_type\": \"Markdown\"}\n",
                    "\n",
                    "# Test EasyOCR\n",
                    "print(\"Running EasyOCR (scanned fallback)...\")\n",
                    "start = time.time()\n",
                    "try:\n",
                    "    text_easyocr = extract_easyocr(PDF_PATH)\n",
                    "    results[\"EasyOCR\"] = {\n",
                    "        \"time\": time.time() - start,\n",
                    "        \"char_count\": len(text_easyocr),\n",
                    "        \"text\": text_easyocr,\n",
                    "        \"success\": True,\n",
                    "        \"output_type\": \"Plain Text\"\n",
                    "    }\n",
                    "except Exception as e:\n",
                    "    results[\"EasyOCR\"] = {\"time\": 0, \"char_count\": 0, \"text\": \"\", \"success\": False, \"error\": str(e), \"output_type\": \"Plain Text\"}\n",
                    "\n",
                    "print(\"All models executed!\")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 6. Comparison Results"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "df_comparison = pd.DataFrame([\n",
                    "    {\n",
                    "        \"Method\": name,\n",
                    "        \"Execution Time (s)\": f\"{data['time']:.4f}\" if data['success'] else \"Failed\",\n",
                    "        \"Character Count\": data['char_count'] if data['success'] else 0,\n",
                    "        \"Output Format\": data['output_type'],\n",
                    "        \"Status\": \"Success\" if data['success'] else f\"Failed: {data.get('error')}\"\n",
                    "    }\n",
                    "    for name, data in results.items()\n",
                    "])\n",
                    "display(df_comparison)"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 7. Sample Outputs (Snippets)"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "for name, data in results.items():\n",
                    "    if data['success']:\n",
                    "        print(f\"\\n=====================================\")\n",
                    "        print(f\" METHOD: {name} (Snippet)\")\n",
                    "        print(f\"=====================================\")\n",
                    "        snippet = data['text'][:800]\n",
                    "        print(snippet)\n",
                    "        if len(data['text']) > 800:\n",
                    "            print(\"...\\n[TRUNCATED]\")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 8. Save MVP Deliverables"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Select the best plain text content\n",
                    "is_digital = results[\"PyMuPDF\"][\"char_count\"] > 100\n",
                    "best_plain_method = \"PyMuPDF\" if is_digital else \"EasyOCR\"\n",
                    "plain_text_out = results[best_plain_method][\"text\"]\n",
                    "\n",
                    "# Save MVP 1\n",
                    "with open(\"8_operadora_y_desarrolladora_de_industrias_plain.txt\", \"w\", encoding=\"utf-8\") as f:\n",
                    "    f.write(plain_text_out)\n",
                    "print(f\"Saved MVP 1 (Plain Text) to '8_operadora_y_desarrolladora_de_industrias_plain.txt'\")\n",
                    "\n",
                    "# Save MVP 2\n",
                    "best_md_method = \"PyMuPDF4LLM\"\n",
                    "md_text_out = results[best_md_method][\"text\"]\n",
                    "with open(\"8_operadora_y_desarrolladora_de_industrias_markdown.md\", \"w\", encoding=\"utf-8\") as f:\n",
                    "    f.write(md_text_out)\n",
                    "print(f\"Saved MVP 2 (Markdown) to '8_operadora_y_desarrolladora_de_industrias_markdown.md'\")"
                ]
            }
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 2
    }
    
    with open("ocr_evaluation.ipynb", "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=1)
    print("Notebook 'ocr_evaluation.ipynb' created successfully!")

if __name__ == "__main__":
    create_notebook()
