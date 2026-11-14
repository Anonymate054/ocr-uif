# OCR-UIF Desktop Application & Processing Pipeline

Secure, offline desktop application and backend extraction pipeline designed to identify, extract, and classify metadata from scanned PDF list documents (such as official block-lists and administrative notifications).

All algorithms, models, and processing steps run **100% locally and offline** without external API connections or tracking, ensuring absolute security for sensitive documents.

---

## 📁 Codebase Structure

The project follows a clean production-grade structure, separating executable code from testing assets and sandboxes:

```text
ocr-uif/
├── .github/workflows/
│   ├── tests.yml                # Lint + unit tests on every push and PR
│   ├── build.yml                # Builds OCR-UIF.exe on Windows (downloadable artifact)
│   └── tag-release.yml          # Creates the vX.Y.Z tag of every version on merge to master
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
├── scripts/build_exe.py         # Packages the app with flet pack (used by CI too)
├── scripts/bump_version.py      # Bumps VERSION and ui/version.py
├── scripts/tag_versions.py      # Creates missing version tags (run by CI)
├── requirements.txt             # Production dependencies (exact versions)
├── requirements-build.txt       # Packaging tools (PyInstaller, flet-cli, flet-desktop)
└── README.md                    # This guide
```

*Note: All development testing data (`files/`), intermediate OCR transcriptions (`transcriptions/`), and historical research tools/sandboxes (`old/`) are ignored in Git.*

---

## 🛠️ Environment Setup & Local Launch

### 1. Requirements
* Python 3.13 (the version used by CI and the build; dependencies are pinned for it)
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

### Automatic build (recommended)
Every push to `master`, every version tag and every pull request runs `.github/workflows/build.yml` on a Windows runner. Download `OCR-UIF.exe` from the run's **Artifacts** (GitHub → Actions → *build exe*). Copy it to `OCR-UIF-Release/` for distribution.

### Local build
On Windows (or Linux/macOS for a native binary of that system), in a clean virtual environment:

```bash
pip install -r requirements.txt -r requirements-build.txt
python scripts/build_exe.py            # single file: dist/OCR-UIF.exe
python scripts/build_exe.py --onedir   # folder: starts faster
```

`scripts/build_exe.py` runs `flet pack` with the data the app needs (`VERSION`, `models/`, `ui/assets/` and RapidOCR's own package folder, which holds its `config.yaml`) and the hidden imports of the CRF model. It takes every path from the current environment, so it does not depend on a particular machine.

## Versioning and tags

Every change bumps the version with `python scripts/bump_version.py patch|minor|major` (updates `VERSION` and `DEFAULT_VERSION` in `ui/version.py`). When the change is merged into `master`, `.github/workflows/tag-release.yml` runs `scripts/tag_versions.py --push`, which creates the `vX.Y.Z` tag of every version that does not have one yet.

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
  * **Errors and manual review:** A PDF that fails (unreadable file, no text found) gets no CSV; the technical detail goes to `_errores.csv`. Everything a person should check by hand goes to one report, `_revisar.csv` (columns `ARCHIVO`, `MOTIVO_REVISION`): unreadable PDFs, CSVs without a name or office number, doubtful ALTA/BAJA and texts that do not look like a UIF office. When the run ends the app shows only counts (processed, altas, bajas, how many need review, how many could not be read), never file names, and an **Abrir reporte de revisión** button that opens `_revisar.csv` in the default program (Excel). Both reports are removed at the start of each run, so a cancelled run never leaves the previous run's reports behind.
  * **Cache writes are atomic:** each transcription is written to a temporary file and renamed, so closing the app mid-write never leaves a truncated transcription in the cache.

#### 2. Stage 2: Offline NLP Document Classifier (TF-IDF + SVM Classifier)
* **How it works:**
  * **Feature Extraction:** Reconstructed text is processed by a **TF-IDF Vectorizer** (`models/tfidf_vectorizer.joblib`) to extract word and bigram patterns.
  * **Prediction:** A pre-trained **Support Vector Machine (SVM) Classifier** (`models/svm_classifier_model.joblib`) analyzes the text features.
  * **Retraining:** `python scripts/train_classifier.py files` reads every PDF with the app's own OCR (`en_PP-OCRv4`), labels it with the `MOVIMIENTO` of its reference CSV and rewrites the three models in `models/`. Since v2.3.4 the model is trained on the current OCR, which keeps words apart; the previous model had been trained on text with run-together words (1,032 such terms in its vocabulary).
  * **Confidence check (v2.4.0):** every prediction comes with its probability. A CSV whose ALTA/BAJA has less than `MIN_CONFIDENCE` (0.90) is listed in `_revisar.csv` as doubtful, and a text with fewer than `MIN_KNOWN_TERMS` (100) words of the model's vocabulary is listed as "not a UIF office" (the model would still answer ALTA or BAJA for an electricity bill). On the 116 project PDFs the lowest probability is 0.72 (one real doubtful case) and the rest are above 0.95; real offices contain 271 or more known terms.
  * **No personal data in the published model:** `models/` is public, so the vocabulary keeps only terms that appear in at least `MIN_DOC_FREQ` (3) offices, never terms with digits (RFC, office numbers, dates) and never words of the names of the people or companies used for training (`safe_vocabulary` in `ui/pipeline.py`, `name_words` in `scripts/train_classifier.py`). With that vocabulary (816 terms) the 5-fold cross-validation stays at 100 %.
  * **Measured results (v2.3.5, 14 Nov 2026):** with the 116 project PDFs, compared against the reference CSVs that could be paired (95): MOVIMIENTO 95/95, office number 95/95, full row 93/95, names with Ñ 7/7. The cross-validation table below was measured on 115 documents from a single issuer, so it does not guarantee the same accuracy on new formats; always review the BAJA results.

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
  * **Segmentation (`ui/ner_segmenter.py`):** The CRF tagger splits people into NOMBRE / PATERNO / MATERNO and detects companies. The CRF tags almost any unknown 2-3 word name as a company, so a name is only treated as a company when it also contains a company word (`SA`, `SC`, `GRUPO`, `ASOCIACION`...); otherwise it is split by position (`XIULAN MEI → XIULAN / MEI`). Extracted names are **not** matched against other CSV files on disk.
  * **CSV output:** One CSV per PDF, encoded in Windows-1252 (same as the reference CSVs) so Excel shows `Ñ` correctly.

---

## 🔒 Security & Offline Design
1. **Model Cache:** All models (OCR ONNX engines, TF-IDF vectorizer, and SVM classifier weights) are bundled locally in the `/models` directory or cached on first launch.
2. **Zero Outbound Traffic:** No telemetry, tracking, or network requests are executed at runtime. The application itself never opens a network connection (the packaged Flet runtime includes networking libraries, but the app does not use them).
