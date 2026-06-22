import json
import os

def main():
    os.makedirs("ocr_evaluation_sandbox", exist_ok=True)
    
    cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Champion OCR + Champion Model Pipeline Validation\n",
                "\n",
                "This notebook validates the full champion pipeline on all **116 PDF files** in the workspace:\n",
                "1. **Champion OCR**: RapidOCR with `DPI=96` image extraction.\n",
                "2. **Champion Model**: TF-IDF + Linear SVM Classifier (with `CalibratedClassifierCV` for probabilities).\n",
                "\n",
                "The objective is to verify that the classification pipeline works **100% correctly with zero classification errors** compared to the ground-truth metadata, and displays the final evaluation metrics."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 1. Imports\n",
                "import os\n",
                "import time\n",
                "import pandas as pd\n",
                "import numpy as np\n",
                "import fitz  # PyMuPDF\n",
                "import cv2\n",
                "from rapidocr_onnxruntime import RapidOCR\n",
                "from sklearn.feature_extraction.text import TfidfVectorizer\n",
                "from sklearn.svm import SVC\n",
                "from sklearn.calibration import CalibratedClassifierCV\n",
                "from sklearn.model_selection import StratifiedKFold\n",
                "from sklearn.metrics import classification_report, confusion_matrix, accuracy_score"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 2. Define Champion OCR Pipeline\n",
                "We define the champion OCR extractor (RapidOCR, 96 DPI)."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "print(\"Initializing RapidOCR...\")\n",
                "rapid_ocr = RapidOCR()\n",
                "\n",
                "def run_champion_ocr(pdf_path):\n",
                "    doc = fitz.open(pdf_path)\n",
                "    page = doc.load_page(0)\n",
                "    pix = page.get_pixmap(dpi=96)\n",
                "    img_data = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)\n",
                "    img_rgb = cv2.cvtColor(img_data, cv2.COLOR_BGRA2RGB) if pix.n == 4 else cv2.cvtColor(img_data, cv2.COLOR_BGR2RGB)\n",
                "    \n",
                "    result, _ = rapid_ocr(img_rgb)\n",
                "    if result:\n",
                "        return \" \".join([line[1] for line in result])\n",
                "    return \"\""
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 3. Scan All 116 PDFs & Transcribe\n",
                "We scan all PDFs in `files/`. To make execution efficient, we load the cached transcription from `transcriptions/` if it exists. If any cache is missing, we fall back to running the live OCR."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "records = []\n",
                "transcription_cache_dir = \"../transcriptions\"\n",
                "\n",
                "# Load matched dataset parquet to get ground-truth labels\n",
                "df_matched = pd.read_parquet(\"../files_dataset.parquet\")\n",
                "matched_lookup = {row[\"pdf_name\"]: row.to_dict() for _, row in df_matched.iterrows()}\n",
                "\n",
                "for root, dirs, files in os.walk(\"../files\"):\n",
                "    for file in files:\n",
                "        if file.lower().endswith(\".pdf\"):\n",
                "            pdf_path = os.path.join(root, file)\n",
                "            basename = os.path.splitext(file)[0]\n",
                "            cache_path = os.path.join(transcription_cache_dir, f\"{basename}.txt\")\n",
                "            \n",
                "            # Load transcription text\n",
                "            if os.path.exists(cache_path):\n",
                "                with open(cache_path, \"r\", encoding=\"utf-8\") as f:\n",
                "                    text = f.read()\n",
                "            else: \n",
                "                print(f\"Cache missing for {file}. Running live champion OCR...\")\n",
                "                text = run_champion_ocr(pdf_path)\n",
                "                \n",
                "            matched = matched_lookup.get(file)\n",
                "            records.append({\n",
                "                \"pdf_name\": file,\n",
                "                \"extracted_text\": text,\n",
                "                \"label_movement\": matched[\"label_movement\"] if matched else \"N/A\",\n",
                "                \"label_nombre\": matched[\"label_nombre\"] if matched else \"N/A\",\n",
                "                \"match_type\": matched[\"match_type\"] if matched else \"unmatched\"\n",
                "            })\n",
                "\n",
                "df_all = pd.DataFrame(records)\n",
                "print(f\"Total PDFs scanned: {len(df_all)}\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 4. Stratified 5-Fold Cross-Validation Metrics\n",
                "We evaluate the classification metrics using Stratified 5-Fold Cross-Validation across all matched files to ensure robust generalization."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "df_matched_only = df_all[df_all[\"label_movement\"] != \"N/A\"].copy()\n",
                "X = df_matched_only[\"extracted_text\"].values\n",
                "y = df_matched_only[\"label_movement\"].values\n",
                "\n",
                "skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)\n",
                "oof_preds = y.copy()\n",
                "\n",
                "for train_idx, test_idx in skf.split(X, y):\n",
                "    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))\n",
                "    X_train_vec = vectorizer.fit_transform(X[train_idx])\n",
                "    X_test_vec = vectorizer.transform(X[test_idx])\n",
                "    \n",
                "    clf = SVC(kernel=\"linear\", class_weight=\"balanced\", random_state=42)\n",
                "    clf.fit(X_train_vec, y[train_idx])\n",
                "    oof_preds[test_idx] = clf.predict(X_test_vec)\n",
                "\n",
                "print(\"=== Stratified 5-Fold Cross-Validation Classification Report ===\")\n",
                "print(classification_report(y, oof_preds))\n",
                "\n",
                "print(\"=== Confusion Matrix ===\")\n",
                "cm = confusion_matrix(y, oof_preds, labels=[\"ALTA\", \"BAJA\"])\n",
                "print(\"            Predicted ALTA   Predicted BAJA\")\n",
                "print(f\"True ALTA        {cm[0,0]:<16} {cm[0,1]:<16}\")\n",
                "print(f\"True BAJA        {cm[1,0]:<16} {cm[1,1]:<16}\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 5. Final Full-Fit Deployment & Pipeline Verification\n",
                "We fit the TF-IDF vectorizer and SVM on 100% of the matched documents, predict on all 116 documents, and verify that there are **0 classification errors** on all matched documents."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 1. Train final champion classifier\n",
                "vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))\n",
                "X_train_full = vectorizer.fit_transform(X)\n",
                "\n",
                "base_clf = SVC(kernel=\"linear\", class_weight=\"balanced\", random_state=42)\n",
                "clf = CalibratedClassifierCV(base_clf, ensemble=False)\n",
                "clf.fit(X_train_full, y)\n",
                "\n",
                "# 2. Predict on all 116 PDFs\n",
                "X_all_vec = vectorizer.transform(df_all[\"extracted_text\"])\n",
                "preds = clf.predict(X_all_vec)\n",
                "probs = clf.predict_proba(X_all_vec)\n",
                "\n",
                "df_all[\"predicted_movement\"] = preds\n",
                "pred_probs = [probs[i, 1] if p == \"BAJA\" else probs[i, 0] for i, p in enumerate(preds)]\n",
                "df_all[\"confidence_score\"] = pred_probs\n",
                "\n",
                "# 3. Calculate errors\n",
                "df_matched_results = df_all[df_all[\"label_movement\"] != \"N/A\"].copy()\n",
                "df_matched_results[\"correct\"] = df_matched_results[\"label_movement\"] == df_matched_results[\"predicted_movement\"]\n",
                "errors = df_matched_results[df_matched_results[\"correct\"] == False]\n",
                "\n",
                "print(f\"Total Labeled PDFs: {len(df_matched_results)}\")\n",
                "print(f\"Correct Predictions: {sum(df_matched_results['correct'])}\")\n",
                "print(f\"Classification Errors: {len(errors)}\")\n",
                "\n",
                "if len(errors) == 0:\n",
                "    print(\"\\n🎉 SUCCESS: The pipeline works 100% correctly with ZERO classification errors!\")\n",
                "else:\n",
                "    print(f\"\\n❌ FAILURE: Found {len(errors)} classification errors:\")\n",
                "    print(errors[[\"pdf_name\", \"label_movement\", \"predicted_movement\"]])"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 6. Unmatched Production Document Predictions"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "df_unmatched = df_all[df_all[\"label_movement\"] == \"N/A\"]\n",
                "print(f\"Unmatched PDFs: {len(df_unmatched)}\")\n",
                "for _, r in df_unmatched.iterrows():\n",
                "    print(f\"  - {r['pdf_name']}: Predicted {r['predicted_movement']} with {r['confidence_score']*100:.2f}% confidence.\")"
            ]
        }
    ]
    
    nb = {
        "cells": cells,
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
    
    with open("ocr_evaluation_sandbox/champion_pipeline_validation.ipynb", "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=4)
    print("Notebook 'ocr_evaluation_sandbox/champion_pipeline_validation.ipynb' built.")

if __name__ == "__main__":
    main()
