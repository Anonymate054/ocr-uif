import json
import os

def build_dataset_generator_nb():
    cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Local OCR & Data Extraction Pipeline\n",
                "\n",
                "This notebook implements and demonstrates the document processing pipeline. It aligns scanned PDFs with their respective metadata CSVs, runs OCR (RapidOCR/EasyOCR) on the first page, extracts entities (movement type, names, office numbers), and builds a unified Parquet dataset.\n",
                "\n",
                "---"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 1. Imports and environment setup\n",
                "import os\n",
                "import re\n",
                "import csv\n",
                "import pandas as pd\n",
                "import numpy as np\n",
                "import cv2\n",
                "import fitz\n",
                "from rapidocr_onnxruntime import RapidOCR"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 2. Alignment Logic\n",
                "We define a robust mapping function to match PDFs to their CSV metadata files. We use three alignment methods:\n",
                "1. **Filename matching**: Direct substring overlap within the same directory.\n",
                "2. **Office number in filename**: Regex matching the office number in the PDF filename to the CSV `MOTIVO` field.\n",
                "3. **Office number in OCR text**: Matching the office number extracted from the first page text to the CSV."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "def clean_name(name):\n",
                "    name = name.lower()\n",
                "    name = re.sub(r\"^(10_|11_|12_|13_|14_|15_|16_|17_|18_|19_|20_|21_|22_|23_|24_|25_|1_|2_|3_|4_|5_|6_|7_|8_|9_)\", \"\", name)\n",
                "    name = re.sub(r\"(_cnbv|_vs|\\.pdf|\\.csv)\", \"\", name)\n",
                "    name = name.replace(\"á\", \"a\").replace(\"é\", \"e\").replace(\"í\", \"i\").replace(\"ó\", \"o\").replace(\"ú\", \"u\").replace(\"ñ\", \"n\")\n",
                "    return re.sub(r\"[\\_\\-\\s]+\", \"\", name)\n",
                "\n",
                "def extract_office_number(text):\n",
                "    # Matches office format: 110/K/2924/2026 or 110-G-1329-2026\n",
                "    m = re.search(r\"110[/\\-\\s_]?[GKgk][/\\-\\s_]?\\d+[/\\-\\s_]?\\d+\", text)\n",
                "    if m:\n",
                "        return re.sub(r\"[/\\-\\s_]+\", \"_\", m.group(0).lower())\n",
                "    return None"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 3. Rule-Based Extractor (Regex)\n",
                "We implement a parser to extract details directly from the OCR text of the first page to evaluate how well simple rules perform compared to machine learning. We target:\n",
                "- **Movement type**: `ALTA` (added to list) vs `BAJA` (removed/suspended from list).\n",
                "- **Office number**: Matches the pattern `110/K/...` or `110/G/...`.\n",
                "- **Entity names**: Matches uppercase words following keywords like \"Se elimina... a:\" or \"Lista de Personas Bloqueadas a:\"."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "def rule_based_extract(text):\n",
                "    clean_txt = \" \".join(text.split())\n",
                "    \n",
                "    # 1. Extract Movement\n",
                "    movement = \"UNKNOWN\"\n",
                "    # Baja keywords\n",
                "    if any(kw in clean_txt.lower() for kw in [\"elimina provisionalmente\", \"suspension definitiva\", \"dejar sin efecto\"]):\n",
                "        movement = \"BAJA\"\n",
                "    # Alta keywords\n",
                "    elif any(kw in clean_txt.lower() for kw in [\"actualizacion de la lista\", \"personas bloqueadas\", \"adicionar\"]):\n",
                "        # Double check it is not a Baja referencing an Alta\n",
                "        if not any(kw in clean_txt.lower() for kw in [\"elimina provisionalmente\", \"suspension definitiva\"]):\n",
                "            movement = \"ALTA\"\n",
                "            \n",
                "    # 2. Extract Office Number (Motivo)\n",
                "    office = \"\"\n",
                "    m_off = re.search(r\"Oficio\\s*No\\.?\\s*(110[/\\-\\s]?[GKgk][/\\-\\s]?\\d+[/\\-\\s]?\\d+)\", clean_txt, re.IGNORECASE)\n",
                "    if m_off:\n",
                "        office = \"OFICIO NO. \" + m_off.group(1).upper()\n",
                "        \n",
                "    return {\n",
                "        \"extracted_movement\": movement,\n",
                "        \"extracted_office\": office\n",
                "    }"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 4. Load and Validate the Generated Parquet Dataset\n",
                "The background script `build_dataset.py` processes all PDFs and CSVs and outputs `files_dataset.parquet`. Let's load it and inspect its properties."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Load the dataset\n",
                "parquet_path = \"files_dataset.parquet\"\n",
                "if os.path.exists(parquet_path):\n",
                "    df = pd.read_parquet(parquet_path)\n",
                "    print(f\"Parquet dataset loaded successfully! Shape: {df.shape}\")\n",
                "    \n",
                "    # Value counts\n",
                "    print(\"\\nMovement Label Distribution:\")\n",
                "    print(df[\"label_movement\"].value_counts())\n",
                "    \n",
                "    # Sample records\n",
                "    display(df[[\"pdf_name\", \"label_movement\", \"label_nombre\", \"match_type\"]].head(5))\n",
                "else:\n",
                "    print(\"files_dataset.parquet not found yet. Make sure build_dataset.py has finished running.\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 5. Evaluate Name and Rule-Based Extractions\n",
                "Let's run our rule-based parser on all documents and evaluate its accuracy against the ground truth labels."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "if os.path.exists(parquet_path):\n",
                "    parsed_results = []\n",
                "    for idx, row in df.iterrows():\n",
                "        extracted = rule_based_extract(row[\"extracted_text\"])\n",
                "        parsed_results.append(extracted)\n",
                "        \n",
                "    df_parsed = pd.DataFrame(parsed_results)\n",
                "    df_eval = pd.concat([df, df_parsed], axis=1)\n",
                "    \n",
                "    # 1. Evaluate Rule-based Movement Classification\n",
                "    correct_movement = (df_eval[\"label_movement\"] == df_eval[\"extracted_movement\"]).sum()\n",
                "    accuracy_movement = (correct_movement / len(df_eval)) * 100\n",
                "    print(f\"Rule-Based Movement Classification Accuracy: {accuracy_movement:.2f}% ({correct_movement}/{len(df_eval)})\")\n",
                "    \n",
                "    # 2. Evaluate Office Number Extraction\n",
                "    # Normalize spaces and compare\n",
                "    def norm_motivo(val):\n",
                "        return re.sub(r\"[^a-zA-Z0-9]\", \"\", str(val).lower())\n",
                "    \n",
                "    correct_office = df_eval.apply(lambda r: norm_motivo(r[\"label_motivo\"]) in norm_motivo(r[\"extracted_office\"]), axis=1).sum()\n",
                "    accuracy_office = (correct_office / len(df_eval)) * 100\n",
                "    print(f\"Rule-Based Office Number Extraction Accuracy: {accuracy_office:.2f}% ({correct_office}/{len(df_eval)})\")\n",
                "else:\n",
                "    print(\"Parquet dataset not loaded.\")"
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
    
    with open("ocr_dataset_generator.ipynb", "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=4)
    print("Notebook 'ocr_dataset_generator.ipynb' built.")

def build_classifier_sandbox_nb():
    cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Movement Classification: Stratified 5-Fold Cross-Validation Sandbox\n",
                "\n",
                "This notebook performs a rigorous **Stratified 5-Fold Cross-Validation** evaluation across all 115 matched transcription files in `transcriptions/`.\n",
                "\n",
                "Using cross-validation ensures that every document is used for both training and validation, producing highly reliable, stable, and unbiased performance metrics. We compare:\n",
                "1. **Rule-Based Regex Classifier**\n",
                "2. **TF-IDF + Naive Bayes**\n",
                "3. **TF-IDF + Logistic Regression**\n",
                "4. **TF-IDF + Support Vector Machine (Linear SVM)**\n",
                "5. **HuggingFace Semantic Embeddings (Sentence Transformers) + Linear SVM**\n",
                "\n",
                "---"
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
                "from sklearn.feature_extraction.text import TfidfVectorizer\n",
                "from sklearn.naive_bayes import MultinomialNB\n",
                "from sklearn.linear_model import LogisticRegression\n",
                "from sklearn.svm import SVC\n",
                "from sklearn.model_selection import StratifiedKFold\n",
                "from sklearn.metrics import accuracy_score, f1_score, classification_report\n",
                "from sentence_transformers import SentenceTransformer"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 2. Load Dataset\n",
                "We load all 115 matched documents from the Parquet dataset generated by the OCR mapping pipeline."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "parquet_path = \"files_dataset.parquet\"\n",
                "if not os.path.exists(parquet_path):\n",
                "    raise FileNotFoundError(\"files_dataset.parquet is missing. Run build_dataset.py first!\")\n",
                "    \n",
                "df = pd.read_parquet(parquet_path)\n",
                "print(f\"Loaded dataset with {len(df)} aligned documents.\")\n",
                "print(\"\\nMovement Label Distribution:\")\n",
                "print(df[\"label_movement\"].value_counts())\n",
                "\n",
                "X = df[\"extracted_text\"].values\n",
                "y = df[\"label_movement\"].values"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 3. Setup Stratified 5-Fold Cross-Validation\n",
                "We initialize a Stratified 5-Fold split (shuffled with a fixed seed for exact reproducibility)."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "n_splits = 5\n",
                "skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)\n",
                "results_dict = {}\n",
                "\n",
                "def run_cv(model_func, is_tfidf=False, X_emb=None):\n",
                "    accs = []\n",
                "    f1s = []\n",
                "    times = []\n",
                "    \n",
                "    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):\n",
                "        start_time = time.time()\n",
                "        \n",
                "        if X_emb is not None:\n",
                "            # Embedding model\n",
                "            X_train_fold, X_test_fold = X_emb[train_idx], X_emb[test_idx]\n",
                "            y_train_fold, y_test_fold = y[train_idx], y[test_idx]\n",
                "            \n",
                "            clf = model_func()\n",
                "            clf.fit(X_train_fold, y_train_fold)\n",
                "            preds = clf.predict(X_test_fold)\n",
                "            \n",
                "        elif is_tfidf:\n",
                "            # TF-IDF model\n",
                "            X_train_fold, X_test_fold = X[train_idx], X[test_idx]\n",
                "            y_train_fold, y_test_fold = y[train_idx], y[test_idx]\n",
                "            \n",
                "            vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))\n",
                "            X_train_vec = vectorizer.fit_transform(X_train_fold)\n",
                "            X_test_vec = vectorizer.transform(X_test_fold)\n",
                "            \n",
                "            clf = model_func()\n",
                "            clf.fit(X_train_vec, y_train_fold)\n",
                "            preds = clf.predict(X_test_vec)\n",
                "            \n",
                "        else:\n",
                "            # Rule-based model\n",
                "            X_test_fold = X[test_idx]\n",
                "            y_test_fold = y[test_idx]\n",
                "            preds = [model_func(txt) for txt in X_test_fold]\n",
                "            \n",
                "        elapsed = time.time() - start_time\n",
                "        accs.append(accuracy_score(y_test_fold, preds))\n",
                "        f1s.append(f1_score(y_test_fold, preds, average=\"macro\"))\n",
                "        times.append(elapsed)\n",
                "        \n",
                "    return {\n",
                "        \"acc_mean\": np.mean(accs) * 100,\n",
                "        \"acc_std\": np.std(accs) * 100,\n",
                "        \"f1_mean\": np.mean(f1s),\n",
                "        \"f1_std\": np.std(f1s),\n",
                "        \"time_mean\": np.mean(times)\n",
                "    }"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 4. Evaluate Rule-Based Classifier"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "def rule_based_classify(text):\n",
                "    txt = text.lower()\n",
                "    if any(kw in txt for kw in [\"elimina provisionalmente\", \"suspension definitiva\", \"dejar sin efecto\"]):\n",
                "        return \"BAJA\"\n",
                "    return \"ALTA\"\n",
                "\n",
                "results_dict[\"Rule-Based Regex\"] = run_cv(rule_based_classify, is_tfidf=False)\n",
                "print(\"Rule-Based Regex CV Completed.\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 5. Evaluate Traditional ML Models (TF-IDF)"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# A. Naive Bayes\n",
                "results_dict[\"TF-IDF + Naive Bayes\"] = run_cv(lambda: MultinomialNB(), is_tfidf=True)\n",
                "\n",
                "# B. Logistic Regression\n",
                "results_dict[\"TF-IDF + Logistic Regression\"] = run_cv(\n",
                "    lambda: LogisticRegression(class_weight=\"balanced\", random_state=42), \n",
                "    is_tfidf=True\n",
                ")\n",
                "\n",
                "# C. Linear SVM\n",
                "results_dict[\"TF-IDF + Linear SVM\"] = run_cv(\n",
                "    lambda: SVC(kernel=\"linear\", class_weight=\"balanced\", random_state=42), \n",
                "    is_tfidf=True\n",
                ")\n",
                "print(\"TF-IDF Classifiers CV Completed.\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 6. Evaluate Advanced Semantic Model (Sentence Transformers)"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "print(\"Loading local Sentence Transformer model...\")\n",
                "model_name = \"local_models/paraphrase-multilingual-MiniLM-L12-v2\"\n",
                "if not os.path.exists(model_name):\n",
                "    model_name = \"paraphrase-multilingual-MiniLM-L12-v2\"\n",
                "embedder = SentenceTransformer(model_name)\n",
                "\n",
                "print(\"Generating dense embeddings for all 115 documents...\")\n",
                "X_emb = embedder.encode(df[\"extracted_text\"].tolist(), show_progress_bar=True)\n",
                "\n",
                "results_dict[\"Semantic Embeddings + SVM\"] = run_cv(\n",
                "    lambda: SVC(kernel=\"linear\", class_weight=\"balanced\", random_state=42), \n",
                "    X_emb=X_emb\n",
                ")\n",
                "print(\"Semantic Embeddings SVM CV Completed.\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 7. Compare Cross-Validation Performance"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "cv_results = []\n",
                "for model_name, metrics in results_dict.items():\n",
                "    cv_results.append({\n",
                "        \"Model\": model_name,\n",
                "        \"Mean Accuracy (%)\": f\"{metrics['acc_mean']:.2f}% ± {metrics['acc_std']:.2f}%\",\n",
                "        \"Mean F1 (Macro)\": f\"{metrics['f1_mean']:.4f} ± {metrics['f1_std']:.4f}\",\n",
                "        \"Avg Fit Time (s)\": f\"{metrics['time_mean']:.4f}s\"\n",
                "    })\n",
                "    \n",
                "df_results = pd.DataFrame(cv_results)\n",
                "display(df_results)"
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
    
    with open("movement_classifier_sandbox.ipynb", "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=4)
    print("Notebook 'movement_classifier_sandbox.ipynb' built.")

if __name__ == "__main__":
    build_dataset_generator_nb()
    build_classifier_sandbox_nb()
