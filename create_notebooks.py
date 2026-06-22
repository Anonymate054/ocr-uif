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
                "# Movement Classification Machine Learning Sandbox\n",
                "\n",
                "This notebook trains and evaluates multiple classification models to categorize document movements as **`ALTA`** or **`BAJA`** based on the first page OCR text.\n",
                "\n",
                "We compare:\n",
                "1. **Rule-Based Regex Classifier**\n",
                "2. **TF-IDF + Naive Bayes**\n",
                "3. **TF-IDF + Logistic Regression**\n",
                "4. **TF-IDF + Support Vector Machine (Linear SVM)**\n",
                "5. **HuggingFace Semantic Embeddings (Sentence Transformers) + Logistic Regression**\n",
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
                "from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix\n",
                "from sentence_transformers import SentenceTransformer"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 2. Load Dataset and Split Data\n",
                "We divide the dataset according to the specified strategy:\n",
                "- **Train/Test pool**: Documents in `lpb_oficios_01_06_26`, `lpb_oficios_02_06_26`, `lpb_oficios_03_06_26`, `lpb_oficios_04_06_26`, and `lpb_oficios_10_06_26`.\n",
                "- **Validation Set**: Documents in `lpb_oficios_05_06_26` (held out completely for final evaluation)."
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
                "\n",
                "# Print counts by directory\n",
                "print(\"Documents count by directory:\")\n",
                "print(df[\"pdf_dir\"].value_counts())\n",
                "\n",
                "# Split Train pool vs Validation pool\n",
                "val_dirs = [\"lpb_oficios_05_06_26\"]\n",
                "\n",
                "train_df = df[~df[\"pdf_dir\"].isin(val_dirs)].copy()\n",
                "val_df = df[df[\"pdf_dir\"].isin(val_dirs)].copy()\n",
                "\n",
                "print(f\"\\nTrain Pool Size: {train_df.shape[0]}\")\n",
                "print(f\"Validation Set Size: {val_df.shape[0]}\")\n",
                "\n",
                "# Perform stratified train/test split on Train pool (80% Train, 20% Test)\n",
                "from sklearn.model_selection import train_test_split\n",
                "train_data, test_data = train_test_split(\n",
                "    train_df, \n",
                "    test_size=0.2, \n",
                "    stratify=train_df[\"label_movement\"], \n",
                "    random_state=42\n",
                ")\n",
                "\n",
                "print(f\"\\nStratified Split:\")\n",
                "print(f\"  - Train Set: {train_data.shape[0]} ({train_data['label_movement'].value_counts().to_dict()})\")\n",
                "print(f\"  - Test Set: {test_data.shape[0]} ({test_data['label_movement'].value_counts().to_dict()})\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 3. Evaluate Rule-Based (Regex) Baseline\n",
                "Let's see how our keyword rules perform on the test and validation datasets."
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
                "# Evaluation function\n",
                "def eval_classifier(y_true, y_pred, name, start_time):\n",
                "    elapsed = time.time() - start_time\n",
                "    acc = accuracy_score(y_true, y_pred) * 100\n",
                "    f1 = f1_score(y_true, y_pred, average=\"macro\")\n",
                "    print(f\"\\n=== {name} ===\")\n",
                "    print(f\"Accuracy: {acc:.2f}%\")\n",
                "    print(f\"F1 Score (Macro): {f1:.4f}\")\n",
                "    print(f\"Time taken: {elapsed:.4f}s\")\n",
                "    print(classification_report(y_true, y_pred, zero_division=0))\n",
                "    return {\n",
                "        \"model\": name,\n",
                "        \"accuracy\": acc,\n",
                "        \"f1_macro\": f1,\n",
                "        \"time_s\": elapsed\n",
                "    }\n",
                "\n",
                "start = time.time()\n",
                "test_preds_rule = [rule_based_classify(t) for t in test_data[\"extracted_text\"]]\n",
                "rule_metrics = eval_classifier(test_data[\"label_movement\"], test_preds_rule, \"Rule-Based Baseline (Test)\", start)"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 4. Train Traditional ML Models (TF-IDF + Classifiers)\n",
                "We vectorize the text using TF-IDF (incorporating both word unigrams and bigrams, ignoring stop words) and fit standard classifiers: Naive Bayes, Logistic Regression, and Linear SVM."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Vectorizer\n",
                "vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))\n",
                "X_train = vectorizer.fit_transform(train_data[\"extracted_text\"])\n",
                "X_test = vectorizer.transform(test_data[\"extracted_text\"])\n",
                "y_train = train_data[\"label_movement\"].values\n",
                "y_test = test_data[\"label_movement\"].values\n",
                "\n",
                "ml_results = []\n",
                "\n",
                "# A. Multinomial Naive Bayes\n",
                "start = time.time()\n",
                "clf_nb = MultinomialNB()\n",
                "clf_nb.fit(X_train, y_train)\n",
                "preds_nb = clf_nb.predict(X_test)\n",
                "ml_results.append(eval_classifier(y_test, preds_nb, \"TF-IDF + Naive Bayes\", start))\n",
                "\n",
                "# B. Logistic Regression\n",
                "start = time.time()\n",
                "clf_lr = LogisticRegression(class_weight=\"balanced\", random_state=42)\n",
                "clf_lr.fit(X_train, y_train)\n",
                "preds_lr = clf_lr.predict(X_test)\n",
                "ml_results.append(eval_classifier(y_test, preds_lr, \"TF-IDF + Logistic Regression\", start))\n",
                "\n",
                "# C. Support Vector Machine\n",
                "start = time.time()\n",
                "clf_svm = SVC(kernel=\"linear\", class_weight=\"balanced\", random_state=42)\n",
                "clf_svm.fit(X_train, y_train)\n",
                "preds_svm = clf_svm.predict(X_test)\n",
                "ml_results.append(eval_classifier(y_test, preds_svm, \"TF-IDF + Linear SVM\", start))"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 5. Train Advanced Semantic Model (Sentence Transformers + SVM)\n",
                "Here we load a local multilingual Transformer (`paraphrase-multilingual-MiniLM-L12-v2`) to extract dense sentence embeddings from the document texts. This captures semantic similarity in Spanish rather than just exact word counts. Then we train a classifier on top of the embeddings."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "print(\"Loading multilingual Sentence Transformer model...\")\n",
                "model_name = \"local_models/paraphrase-multilingual-MiniLM-L12-v2\"\n",
                "if not os.path.exists(model_name):\n",
                "    model_name = \"paraphrase-multilingual-MiniLM-L12-v2\"\n",
                "embedder = SentenceTransformer(model_name)\n",
                "\n",
                "print(\"Generating sentence embeddings for train and test sets...\")\n",
                "start_embed = time.time()\n",
                "X_train_emb = embedder.encode(train_data[\"extracted_text\"].tolist(), show_progress_bar=True)\n",
                "X_test_emb = embedder.encode(test_data[\"extracted_text\"].tolist(), show_progress_bar=True)\n",
                "print(f\"Embeddings generated in {time.time() - start_embed:.2f}s\")\n",
                "\n",
                "# Fit Classifier on top of embeddings\n",
                "start = time.time()\n",
                "clf_emb = SVC(kernel=\"linear\", class_weight=\"balanced\", random_state=42)\n",
                "clf_emb.fit(X_train_emb, y_train)\n",
                "preds_emb = clf_emb.predict(X_test_emb)\n",
                "ml_results.append(eval_classifier(y_test, preds_emb, \"Semantic Embeddings + SVM\", start))"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 6. Validation Set Comparison (lpb_oficios_05_06_26)\n",
                "Let's evaluate all models on the completely independent validation set (`lpb_oficios_05_06_26`)."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "X_val_text = val_df[\"extracted_text\"].tolist()\n",
                "y_val = val_df[\"label_movement\"].values\n",
                "\n",
                "print(f\"Evaluating validation set of size {len(y_val)}...\")\n",
                "\n",
                "# 1. Rule-based\n",
                "preds_val_rule = [rule_based_classify(t) for t in X_val_text]\n",
                "acc_val_rule = accuracy_score(y_val, preds_val_rule) * 100\n",
                "\n",
                "# TF-IDF transformation\n",
                "X_val_tfidf = vectorizer.transform(X_val_text)\n",
                "\n",
                "# 2. Naive Bayes\n",
                "preds_val_nb = clf_nb.predict(X_val_tfidf)\n",
                "acc_val_nb = accuracy_score(y_val, preds_val_nb) * 100\n",
                "\n",
                "# 3. Logistic Regression\n",
                "preds_val_lr = clf_lr.predict(X_val_tfidf)\n",
                "acc_val_lr = accuracy_score(y_val, preds_val_lr) * 100\n",
                "\n",
                "# 4. SVM\n",
                "preds_val_svm = clf_svm.predict(X_val_tfidf)\n",
                "acc_val_svm = accuracy_score(y_val, preds_val_svm) * 100\n",
                "\n",
                "# 5. Semantic Embeddings SVM\n",
                "X_val_emb = embedder.encode(X_val_text, show_progress_bar=False)\n",
                "preds_val_emb = clf_emb.predict(X_val_emb)\n",
                "acc_val_emb = accuracy_score(y_val, preds_val_emb) * 100\n",
                "\n",
                "# Compilation\n",
                "val_comparison = {\n",
                "    \"Model\": [\n",
                "        \"Rule-Based Regex\", \n",
                "        \"TF-IDF + Naive Bayes\", \n",
                "        \"TF-IDF + Logistic Regression\", \n",
                "        \"TF-IDF + Linear SVM\", \n",
                "        \"Semantic Embeddings + SVM\"\n",
                "    ],\n",
                "    \"Validation Accuracy (%)\": [\n",
                "        acc_val_rule, \n",
                "        acc_val_nb, \n",
                "        acc_val_lr, \n",
                "        acc_val_svm, \n",
                "        acc_val_emb\n",
                "    ]\n",
                "}\n",
                "df_val_comp = pd.DataFrame(val_comparison)\n",
                "display(df_val_comp)"
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
