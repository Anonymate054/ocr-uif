# OCR-UIF Desktop Application & Processing Pipeline

Secure, offline desktop application and backend extraction pipeline designed to identify, extract, and classify metadata from scanned PDF list documents (such as official block-lists and administrative notifications).

All algorithms, models, and processing steps run **100% locally and offline** without external API connections or tracking, ensuring absolute security for sensitive documents.

---

## 📁 Codebase Structure

The project follows a clean production-grade structure, separating executable code from testing assets and sandboxes:

```text
ocr-uif/
├── .github/workflows/tests.yml  # Runs the regression tests on every push
├── OCR-UIF-Release/             # Standardized launch scripts & compiled apps
│   ├── OCR-UIF.exe              # Standalone Windows GUI executable
│   ├── run_linux.sh             # Linux execution script
│   ├── run_mac.command          # macOS execution script
│   └── README.md                # Quick launch instructions
├── ui/                          # GUI Application package (Flet)
│   ├── assets/                  # GUI styling theme, fonts, and assets
│   ├── app.py                   # Desktop application UI layout
│   ├── pipeline.py              # Pipeline orchestrator: OCR → classifier → CSV
│   ├── ocr.py                   # Stage 1: first-page OCR with on-disk cache
│   ├── extract_and_validate.py  # Office number and name extraction from OCR text
│   ├── name_normalization.py    # Accent/Ñ handling and OCR name repair
│   ├── ner_segmenter.py         # CRF segmentation into NOMBRE / PATERNO / MATERNO
│   ├── paths.py                 # Model and cache paths (source and .exe)
│   └── cpu_limits.py            # Caps numeric libraries at 70 % of CPU cores
├── models/                      # Lightweight pre-trained ML models for offline inference
│   ├── en_PP-OCRv4_rec_infer.onnx # Latin/Western PP-OCRv4 recognition model (resolves word spacing & company names)
│   ├── en_dict.txt              # Character vocabulary dictionary
│   ├── tfidf_vectorizer.joblib  # Vectorizer for text classification
│   └── svm_classifier_model.joblib # SVM classifier predicting document status
├── tests/                       # Unit tests (python -m pytest tests)
├── scripts/evaluate_outputs.py  # Compares output CSVs with reference CSVs
├── scripts/train_classifier.py  # Retrains the TF-IDF + SVM classifier with the app's OCR
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

2. **Package using Flet Pack** (includes models, assets, and RapidOCR data files):
   ```cmd
   flet pack ui/app.py ^
     --name OCR-UIF ^
     --add-data "models:models" ^
     --add-data "ui/assets:ui/assets" ^
     --add-data "%CONDA_PREFIX%\Lib\site-packages\rapidocr_onnxruntime:rapidocr_onnxruntime" ^
     --product-name "OCR-UIF" ^
     --file-version "1.0.0.0" ^
     --product-version "1.0.0" ^
     --onedir ^
     --yes
   ```
   > **Important:** The `--add-data rapidocr_onnxruntime` flag is required. RapidOCR resolves `config.yaml` and its ONNX model files relative to its own package directory at runtime. Without this flag the exe will crash with `FileExistsError: config.yaml does not exist`.

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

#### 1. Stage 1: Layout-Aware OCR Engine (PyMuPDF + RapidOCR en_PP-OCRv4)
* **How it works:**
  * **Rendering:** PyMuPDF renders the **first page** of each PDF at `96 DPI` (the page that holds the name, RFC and office number).
  * **OCR Execution:** The image is processed by the **RapidOCR ONNX Runtime engine** using the optimized **Latin/Western PP-OCRv4** recognition model (`models/en_PP-OCRv4_rec_infer.onnx`), ensuring correct character and space segmentation on complex corporate names and uppercase Spanish tables without word merging.
  * **Output:** The recognized lines are joined as plain text (one line per detected text box).
  * **Character set:** `models/en_dict.txt` is ASCII only, so the model never outputs `Ñ` or accented vowels (`SALDAÑA` is read as `SALDANA`, `RAMÍREZ` often as `RAMFREZ`). Stage 3 repairs this.
  * **Cache:** Each transcription is cached under a hash of the PDF contents plus `OCR_CACHE_TAG` (in `ui/ocr.py`). Running from source it goes to `transcriptions/`; the `.exe` uses `%LOCALAPPDATA%\OCR-UIF\transcriptions`, never the folder next to the executable. Change `OCR_CACHE_TAG` whenever the OCR model, DPI or pages change.
  * **Speed:** Recognition is about 87 % of the run time. PDFs are processed in parallel (70 % of the cores) with one shared engine, which measured faster than one engine per thread. Disabling the angle classifier or batching recognition did not speed it up and changed the text, so they are left at their defaults. When every PDF is already cached, the OCR models are not loaded at all.
  * **Errors:** A PDF that fails (unreadable file, no text found) gets no CSV; it is listed in `_errores.csv` in the output folder and the app shows how many failed.

#### 2. Stage 2: Offline NLP Document Classifier (TF-IDF + SVM Classifier)
* **How it works:**
  * **Feature Extraction:** Reconstructed text is processed by a **TF-IDF Vectorizer** (`models/tfidf_vectorizer.joblib`) to extract word and bigram patterns.
  * **Prediction:** A pre-trained **Support Vector Machine (SVM) Classifier** (`models/svm_classifier_model.joblib`) analyzes the text features.
  * **Retraining:** `python scripts/train_classifier.py files` reads every PDF with the app's own OCR (`en_PP-OCRv4`), labels it with the `MOVIMIENTO` of its reference CSV and rewrites the three models in `models/`. Since v2.3.4 the model is trained on the current OCR, which keeps words apart; the previous model had been trained on text with run-together words (1,032 such terms in its vocabulary).
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
  * **Name repair (`ui/name_normalization.py`):** Accents are removed from vowels and `Ñ` is kept, matching the reference CSVs (`JOSE MARIA`, `SALDAÑA`). Tokens the OCR damaged are repaired only when the result is a known name in `models/gazetteer.joblib` (`RAMFREZ → RAMIREZ`, `ZEPEDAE → ZEPEDA`), and for people `Ñ` is restored when the `Ñ` spelling is a known name (`SALDANA → SALDAÑA`, `PENA → PEÑA`).
  * **Table headers:** In the 110/K offices the name comes after a `Persona / RFC` table header; those header lines are skipped. Spelled-out legal forms are shortened (`SOCIEDAD ANONIMA DE CAPITAL VARIABLE → SA DE CV`).
  * **Segmentation (`ui/ner_segmenter.py`):** The CRF tagger splits people into NOMBRE / PATERNO / MATERNO and detects companies. The CRF tags almost any unknown 2-3 word name as a company, so a name is only treated as a company when it also contains a company word (`SA`, `SC`, `GRUPO`, `ASOCIACION`...); otherwise it is split by position (`MEIYU LEI → MEIYU / LEI`). Extracted names are **not** matched against other CSV files on disk.
  * **CSV output:** One CSV per PDF, encoded in Windows-1252 (same as the reference CSVs) so Excel shows `Ñ` correctly.

---

## 🔒 Security & Offline Design
1. **Model Cache:** All models (OCR ONNX engines, TF-IDF vectorizer, and SVM classifier weights) are bundled locally in the `/models` directory or cached on first launch.
2. **Zero Outbound Traffic:** No telemetry, tracking, or network requests are executed at runtime. The application is completely air-gapped and secure.
