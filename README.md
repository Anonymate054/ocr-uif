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

## 📦 Compiling to Windows (.exe)

The project includes an automated, offline-ready Windows executable build pipeline. It compiles the Python source code into a single, standalone `OCR-UIF.exe` using a Wine-based Docker container to emulate Windows.

### Prerequisites
* **Docker** installed and running on your system (Linux/macOS/Windows).

### Compilation Options

You can build the executable in two ways: **natively on a Windows machine** (recommended if you have access to one) or **cross-compiling on Linux/macOS** using Docker.

#### Option A: Native Compilation on a Windows Machine (Recommended)
If you are compiling directly on a Windows computer, you do not need Docker or Wine. Open Command Prompt (`cmd`) or PowerShell in the project directory and run:

1. **Install dependencies offline** from the local wheel storage:
   ```cmd
   pip install --no-index --find-links=windows_wheels flet pymupdf rapidocr-onnxruntime scikit-learn pandas numpy joblib opencv-python pyinstaller flet-cli flet-desktop
   ```

2. **Generate the initial Spec configuration** using Flet Pack pointing to the local offline client:
   ```cmd
   set FLET_CLIENT_URL=file:///%CD%/flet-windows.zip
   flet pack ui/app.py --add-data models;models --add-data ui/assets;ui/assets --name OCR-UIF
   ```

3. **Inject the RapidOCR ONNX model files** and Flet assets into the `.spec` file:
   ```cmd
   python modify_spec.py
   ```

4. **Compile the final production standalone executable** with PyInstaller:
   ```cmd
   python -m PyInstaller --clean -y OCR-UIF.spec
   ```

#### Option B: Cross-Compiling on Linux/macOS (via Docker & Wine)
If you are on a Linux or macOS machine, the process is fully automated using Docker:

1. Run the local build script:
   ```bash
   bash build_windows_exe.sh
   ```
2. **What the script does under the hood**:
   * Launches a Docker container running Wine (`mymi14s/ubuntu-wine:24.04-3.11`).
   * Installs Python packages offline using the local dependencies cached in `windows_wheels/`.
   * Packages the initial bundle using `flet pack` and caches Flet's offline client archive (`flet-windows.zip`).
   * Programmatically edits the generated `.spec` file via `modify_spec.py` to inject the RapidOCR ONNX model files and the offline Flet client zip.
   * Compiles the final production standalone executable using PyInstaller.

### Output
The finalized executable is written to `dist/OCR-UIF.exe` in both options. Copy it to the `OCR-UIF-Release/` directory for portable client deployment.

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

### 🏆 Champion Pipeline: Composition & How It Works

The backend (implemented in `ui/pipeline.py`) uses a two-stage hybrid architecture combining high-speed layout-preserving OCR with an offline NLP text classifier.

#### 1. Stage 1: Layout-Aware OCR Engine (PyMuPDF + RapidOCR)
* **How it works:**
  * **Direct Text Extraction:** The pipeline first queries the PDF using PyMuPDF. If the file contains digital selectable text, it is extracted immediately (takes `< 0.05s`).
  * **OCR Fallback:** If the PDF is scanned, PyMuPDF extracts each page as a high-resolution image (`150 DPI`).
  * **OCR Execution:** The image is processed by the **RapidOCR ONNX Runtime engine**.
  * **Layout Reconstruction:** The OCR results are reconstructed into structured markdown text, preserving columns, headers, and table formatting.

#### 2. Stage 2: Offline NLP Document Classifier (TF-IDF + SVM Classifier)
* **How it works:**
  * **Feature Extraction:** Reconstructed text is processed by a **TF-IDF Vectorizer** (`models/tfidf_vectorizer.joblib`) to extract word and bigram patterns.
  * **Prediction:** A pre-trained **Support Vector Machine (SVM) Classifier** (`models/svm_classifier_model.joblib`) analyzes the text features.
  * **Validation Benchmarks:** During test evaluations, the SVM classifier achieved **100% classification accuracy** (perfect precision and recall), ensuring absolute reliability in document status determination.

##### NLP Classification Model Benchmarks & Strategies

To select the most robust classification strategy for identifying register additions (**alta**) versus deregistrations (**baja**), we evaluated five different strategies using a 5-fold cross-validation scheme on our evaluation dataset (115 documents):

| Classification Strategy | Mean CV Accuracy | Mean F1 Score (Macro) | Avg Fit Time (s) | Verdict / Selection |
| :--- | :---: | :---: | :---: | :---: |
| **TF-IDF + Linear SVM** | **100.00% ± 0.00%** | **1.0000 ± 0.0000** | **~0.171s** | 🏆 **Champion (Best precision, fast, stable)** |
| **TF-IDF + Logistic Regression** | 99.13% ± 1.74% | 0.9832 ± 0.0337 | ~0.153s | Runner-Up (Missed 1 "baja" class) |
| **Rule-Based Regex** | 83.48% ± 1.74% | 0.5274 ± 0.0931 | < 0.002s | Poor (Misses context/semantic variations) |
| **TF-IDF + Naive Bayes** | 81.74% ± 1.74% | 0.4497 ± 0.0053 | ~0.103s | Failed (Biased towards majority class) |
| **Semantic Embeddings + SVM** | 70.43% ± 14.39% | 0.5456 ± 0.1362 | ~0.009s | Unstable (High variance on short text blocks) |

##### Confusion Matrices & Error Analysis

* **🏆 TF-IDF + Linear SVM (Selected Champion)**
  Perfect separation of classes; successfully flags all edge cases:
  ```text
                        Predicted ALTA    Predicted BAJA
  Actual ALTA (94)            94                 0
  Actual BAJA (21)             0                21
  ```

* **TF-IDF + Logistic Regression (Runner-Up)**
  Slightly less confident on boundary cases; misclassified one "baja" document as "alta":
  ```text
                        Predicted ALTA    Predicted BAJA
  Actual ALTA (94)            94                 0
  Actual BAJA (21)             1                20
  ```


#### 3. Stage 3: Entity Extractor & Name Standardizer
* **How it works:**
  * **Heuristic Matching:** Regex patterns identify entry numbers, movement dates, and official reference numbers.
  * **Fuzzy Alignment:** Bigram and similarity threshold comparisons match extracted entity names against local records to output clean CSV files.

---

## 🔒 Security & Offline Design
1. **Model Cache:** All models (OCR ONNX engines, TF-IDF vectorizer, and SVM classifier weights) are bundled locally in the `/models` directory or cached on first launch.
2. **Zero Outbound Traffic:** No telemetry, tracking, or network requests are executed at runtime. The application is completely air-gapped and secure.
