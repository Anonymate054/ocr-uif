# Local OCR & PDF Text Extraction Sandbox

This project is a secure, local sandbox environment designed to extract plain text and layout-structured Markdown from PDF documents. Since the documents contain highly sensitive information (CNBV/UIF block lists), **all models and processing pipelines run 100% locally and offline**, without any external API calls or internet dependencies.

---

## 📋 Table of Contents
1. [Project Tree Structure](#-project-tree-structure)
2. [Replicability & Environment Setup](#-replicability--environment-setup)
3. [Sandbox Results (MVPs)](#-sandbox-results-mvps)
4. [Offline Execution Guide](#-offline-execution-guide)
5. [Scripts & Notebook Guide](#-scripts--notebook-guide)

---

## 📁 Project Tree Structure

```text
ocr-uif/
├── .gitignore               # Ignores .venv, checkpoints, raw outputs (.txt, .md)
├── requirements.txt         # Virtual environment requirements
├── rename_files.py          # Script that converted folder names and files to snake_case
├── check_pdf_type.py        # Checks if a PDF has a selectable text layer (digital)
├── test_custom_ocr.py       # Main script executing OCR MVPs and caching model weights
├── ocr_evaluation.ipynb     # Jupyter Notebook comparing models
├── files/                   # Directory containing all files
│   ├── lpb_oficios_01_06_26/
│   │   ├── 8_operadora_y_desarrolladora_de_industrias.pdf (Target PDF)
│   │   └── ... (Other cleaned names)
│   └── ...
├── 8_operadora_y_desarrolladora_de_industrias_plain.txt      # MVP 1 Output (Plain text)
└── 8_operadora_y_desarrolladora_de_industrias_markdown.md    # MVP 2 Output (Markdown text)
```

---

## 🛠️ Replicability & Environment Setup

To recreate this environment on another local system, follow these steps:

### 1. Clone the repository and navigate to it:
```bash
git clone <repository_url> ocr-uif
cd ocr-uif
```

### 2. Create and activate a virtual environment:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies:
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 📊 Sandbox Results (MVPs)

Using the test file `files/lpb_oficios_01_06_26/8_operadora_y_desarrolladora_de_industrias.pdf` (which has 2 pages and is a scanned PDF with 0 extractable digital characters), we benchmarked multiple libraries:

| Extraction Method | Format | Execution Time | Offline Ready? | Layout/Table Support |
| :--- | :--- | :--- | :--- | :--- |
| **PyMuPDF (fitz)** | Plain Text | < 0.05s | Yes (100% Native) | No (Scanned returns empty) |
| **pdfplumber** | Plain Text | < 0.10s | Yes (100% Native) | No (Scanned returns empty) |
| **pypdf** | Plain Text | < 0.05s | Yes (100% Native) | No (Scanned returns empty) |
| **EasyOCR** | Plain Text | ~58.08s | Yes (Cached local model) | Basic block sorting |
| **PyMuPDF4LLM + EasyOCR** | Markdown | ~113.59s | Yes (Cached local model) | **Excellent (Preserves bold text, headers, and full tables)** |

### MVP 1 (Plain Text) Deliverable
- Saved to: [8_operadora_y_desarrolladora_de_industrias_plain.txt](file:///home/lenovo/Documents/projects/ocr-uif/8_operadora_y_desarrolladora_de_industrias_plain.txt)
- Method: Direct EasyOCR reading.
- Quality: Extracted all words accurately including Spanish accents (e.g. *Hacienda*, *INTELIGENCIA*, *PERSONAS BLOQUEADAS*).

### MVP 2 (Markdown) Deliverable
- Saved to: [8_operadora_y_desarrolladora_de_industrias_markdown.md](file:///home/lenovo/Documents/projects/ocr-uif/8_operadora_y_desarrolladora_de_industrias_markdown.md)
- Method: Layout-aware PyMuPDF4LLM with a custom `easyocr` plugin.
- Quality: Beautiful structure mapping, including table extraction:
  ```markdown
  |N:|NOMBRE|FC|
  |---|---|---|
  ||OPERADORA Y DESARROLLADORA DE INDUSTRIAS||
  |1|PODEBI SJE SAPI DE CV|30/11/2023|
  ```

---

## 🔒 Offline Execution Guide

Since the code runs on local machines containing sensitive files, we want to ensure zero internet connection is required after the initial setup.

1. **How models are downloaded:** EasyOCR downloads its detection (CRAFT) and recognition (Latin/English/Spanish) models to the user directory `~/.EasyOCR/model/` during its first execution.
2. **Offline Mode:** Once you run `test_custom_ocr.py` or `ocr_evaluation.ipynb` once with internet, the weights are cached locally. Subsequent runs will use the cache and will **never check or attempt to connect to the internet**, ensuring 100% safety and absolute offline functionality.

---

## 🚀 Scripts & Notebook Guide

- **`rename_files.py`**: Standardizes the directory names and filenames. Run this script once to clean raw file names.
- **`check_pdf_type.py`**: Check if a PDF is digital or scanned.
- **`test_custom_ocr.py`**: Executes the MVP pipeline (both plain text and Markdown) on CPU.
- **`ocr_evaluation.ipynb`**: Interactive notebook running comparisons and showing tables.
