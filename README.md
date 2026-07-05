# OCR-UIF Desktop Application & Processing Pipeline

Secure, offline desktop application and backend extraction pipeline designed to identify, extract, and classify metadata from scanned PDF list documents (such as official block-lists and administrative notifications).

All algorithms, models, and processing steps run **100% locally and offline** without external API connections or tracking, ensuring absolute security for sensitive documents.

---

## 📁 Codebase Structure

The project follows a clean production-grade structure, separating executable code from testing assets and sandboxes:

```text
ocr-uif/
├── .github/                     # Automated CI/CD workflows
├── OCR-UIF-Release/             # Standardized launch scripts & compiled apps
│   ├── OCR-UIF.exe              # Standalone Windows GUI executable
│   ├── run_linux.sh             # Linux execution script
│   ├── run_mac.command          # macOS execution script
│   └── README.md                # Quick launch instructions
├── ui/                          # GUI Application package (Flet)
│   ├── assets/                  # GUI styling theme, fonts, and assets
│   ├── app.py                   # Desktop application UI layout
│   ├── pipeline.py              # Background extraction pipeline orchestrator
│   └── extract_and_validate.py  # Core extraction heuristics & text parsers
├── models/                      # Lightweight pre-trained ML models for offline inference
│   ├── tfidf_vectorizer.joblib  # Vectorizer for text classification
│   └── svm_classifier_model.joblib # SVM classifier predicting document status
├── build_windows_exe.sh         # Packaging script for Wine emulation
├── OCR-UIF.spec                 # PyInstaller spec configuration
├── requirements.txt             # Production dependencies
└── README.md                    # This guide
```

*Note: All development testing data (`files/`), intermediate OCR transcriptions (`transcriptions/`), and historical research tools/sandboxes (`old/`) are ignored in Git.*

---

## 🛠️ Environment Setup & Local Launch

### 1. Requirements
* Python 3.11+
* Operating System: Windows, Linux, or macOS

### 2. Standard Installation
```bash
# Clone the repository
git clone <repository_url> ocr-uif
cd ocr-uif

# Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Launching the App
To run the production application locally via python:
```bash
python -m ui.app
```

---

## 📊 Extraction Sandbox Results (Benchmarked)

Using a sample scanned PDF document (containing 2 pages and 0 digital selectable characters), the extraction pipeline achieved the following performance metrics during benchmarking:

| Extraction Method | Format | Execution Time | Offline Ready? | Layout/Table Support |
| :--- | :--- | :--- | :--- | :--- |
| **PyMuPDF (fitz)** | Plain Text | < 0.05s | Yes | No (Scanned returns empty) |
| **pdfplumber** | Plain Text | < 0.10s | Yes | No (Scanned returns empty) |
| **pypdf** | Plain Text | < 0.05s | Yes | No (Scanned returns empty) |
| **RapidOCR** | Plain Text | ~3.50s | Yes | Basic block alignment |
| **PyMuPDF + RapidOCR** | Markdown | ~4.20s | Yes | **Excellent (Preserves table columns and rows)** |

### Example Layout-Aware Extraction (Anonymized)
```markdown
| N.º | NOMBRE | FECHA |
| :--- | :--- | :--- |
| 1 | ENTIDAD DE PRUEBA S.A. DE C.V. | 30/11/2023 |
```

---

## 🔒 Security & Offline Design
1. **Model Cache:** All models (OCR ONNX engines, TF-IDF vectorizer, and SVM classifier weights) are bundled locally in the `/models` directory or cached on first launch.
2. **Zero Outbound Traffic:** No telemetry, tracking, or network requests are executed at runtime. The application is completely air-gapped and secure.
