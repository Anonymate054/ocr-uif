import os
import json
import time
import joblib
import pandas as pd
from sklearn_crfsuite import CRF
from sklearn_crfsuite import metrics as crf_metrics

LAB_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(LAB_DIR, "dataset")
DATA_DIR = os.path.join(LAB_DIR, "data")
MODELS_DIR = os.path.join(LAB_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

# 1. Load Gazetteers / Dictionaries for Feature Extraction
def load_set(filename):
    p = os.path.join(DATA_DIR, filename)
    if not os.path.exists(p):
        return set()
    with open(p, "r", encoding="utf-8") as f:
        return set(line.strip().upper() for line in f if line.strip())

SURNAMES_SET = load_set("sep_apellidos.txt")
FIRSTNAMES_SET = load_set("sep_hombres.txt").union(load_set("sep_mujeres.txt"))
UIF_CORP_SET = load_set("uif_domain_companies.txt")

PREPOSITIONS = {'DE', 'DEL', 'LA', 'LAS', 'LOS', 'SAN', 'SANTA', 'Y', 'VON', 'VAN'}
CORP_KEYWORDS = {
    'SA', 'CV', 'SC', 'AC', 'SAPI', 'RL', 'SAS', 'SOFOM', 'SNC', 'SDR',
    'SINDICATO', 'ASOCIACION', 'CORPORACION', 'CLINICA', 'CONSTRUCTORA', 'GRUPO',
    'SERVICIOS', 'LOGISTICA', 'PROYECTOS', 'DESARROLLOS', 'EDIFICACIONES',
    'INNOVACION', 'PATRONAL', 'CONFEDERACION', 'JURIDICA', 'JURIDICO',
    'COMPETITIVIDAD', 'PRODUCTIVIDAD', 'INTEGRADORA', 'EMPRESARIAL', 'AMERICANA',
    'GLOBAL', 'MINING', 'ACCESORIOS', 'NOVEDADES', 'ADMINISTRATIVO', 'PLANEACION',
    'LEGAL', 'COMERCIAL', 'DISTRIBUIDORA', 'INDUSTRIA', 'INDUSTRIAS', 'OPERADORA'
}

def word2features(sent, i):
    word = sent[i]
    w_upper = word.upper()
    
    features = {
        'bias': 1.0,
        'word.lower()': word.lower(),
        'word[-3:]': word[-3:],
        'word[-2:]': word[-2:],
        'word[:3]': word[:3],
        'word[:2]': word[:2],
        'word.isupper()': word.isupper(),
        'word.isdigit()': word.isdigit(),
        'is_in_surnames': w_upper in SURNAMES_SET,
        'is_in_firstnames': w_upper in FIRSTNAMES_SET,
        'is_prep': w_upper in PREPOSITIONS,
        'is_corp_kw': w_upper in CORP_KEYWORDS,
        'sent_has_corp_kw': any(w.upper() in CORP_KEYWORDS for w in sent),
        'sent_len': len(sent),
        'length': len(word),
    }
    
    # Previous token features
    if i > 0:
        prev_word = sent[i-1]
        pw_upper = prev_word.upper()
        features.update({
            '-1:word.lower()': prev_word.lower(),
            '-1:word.isupper()': prev_word.isupper(),
            '-1:is_in_surnames': pw_upper in SURNAMES_SET,
            '-1:is_in_firstnames': pw_upper in FIRSTNAMES_SET,
            '-1:is_prep': pw_upper in PREPOSITIONS,
            '-1:is_corp_kw': pw_upper in CORP_KEYWORDS,
        })
    else:
        features['BOS'] = True # Beginning of Sentence

    # Previous 2 tokens
    if i > 1:
        prev2_word = sent[i-2]
        features.update({
            '-2:word.lower()': prev2_word.lower(),
            '-2:is_prep': prev2_word.upper() in PREPOSITIONS,
        })

    # Next token features
    if i < len(sent) - 1:
        next_word = sent[i+1]
        nw_upper = next_word.upper()
        features.update({
            '+1:word.lower()': next_word.lower(),
            '+1:word.isupper()': next_word.isupper(),
            '+1:is_in_surnames': nw_upper in SURNAMES_SET,
            '+1:is_in_firstnames': nw_upper in FIRSTNAMES_SET,
            '+1:is_prep': nw_upper in PREPOSITIONS,
            '+1:is_corp_kw': nw_upper in CORP_KEYWORDS,
        })
    else:
        features['EOS'] = True # End of Sentence

    # Next 2 tokens
    if i < len(sent) - 2:
        next2_word = sent[i+2]
        features.update({
            '+2:word.lower()': next2_word.lower(),
            '+2:is_corp_kw': next2_word.upper() in CORP_KEYWORDS,
        })

    return features

def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]

def load_dataset_file(filename):
    path = os.path.join(DATASET_DIR, filename)
    X = []
    y = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            X.append(item["tokens"])
            y.append(item["labels"])
    return X, y

def decode_bilou_prediction(tokens, predicted_tags):
    """
    Decodes predicted BILOU tags into structured fields:
    (is_company, nombre, paterno, materno)
    """
    first_names = []
    second_names = []
    pat_surnames = []
    mat_surnames = []
    company_tokens = []
    
    for token, tag in zip(tokens, predicted_tags):
        tag_type = tag.split("-")[-1] if "-" in tag else tag
        if tag_type == "COMPANY":
            company_tokens.append(token)
        elif tag_type == "FIRST_NAME":
            first_names.append(token)
        elif tag_type == "SECOND_NAME":
            second_names.append(token)
        elif tag_type == "PAT_SURNAME":
            pat_surnames.append(token)
        elif tag_type == "MAT_SURNAME":
            mat_surnames.append(token)
            
    is_company = len(company_tokens) > (len(first_names) + len(pat_surnames) + len(mat_surnames))
    
    if is_company:
        full_company = " ".join(tokens)
        return True, full_company, "", ""
    else:
        # Group First + Second names into Nombre column
        nombre_parts = first_names + second_names
        nombre = " ".join(nombre_parts)
        paterno = " ".join(pat_surnames)
        materno = " ".join(mat_surnames)
        return False, nombre, paterno, materno

def evaluate_on_real_ground_truth(model_predict_fn):
    """Evaluates the model against real test CSVs from the actual project."""
    test_dirs = [
        r"C:\Users\Anony\Documents\projects\OCR-UIF\test",
        r"C:\Users\Anony\Documents\GitHub\ocr-uif\test"
    ]
    test_dir = next((d for d in test_dirs if os.path.exists(d)), None)
    if not test_dir:
        print("Aviso: No se encontró la carpeta test/ para benchmark real.")
        return 0.0, 0
    
    csv_files = [f for f in os.listdir(test_dir) if f.endswith(".csv")]
    correct_entities = 0
    total = 0
    
    print(f"\n--- Evaluando sobre {len(csv_files)} casos reales de Ground Truth ---")
    for csv_file in csv_files:
        path = os.path.join(test_dir, csv_file)
        try:
            df = pd.read_csv(path)
            if df.empty or "NOMBRE" not in df.columns:
                continue
            
            gt_nom = str(df["NOMBRE"].iloc[0]).strip().upper() if pd.notna(df["NOMBRE"].iloc[0]) else ""
            gt_pat = str(df["PATERNO"].iloc[0]).strip().upper() if "PATERNO" in df.columns and pd.notna(df["PATERNO"].iloc[0]) else ""
            gt_mat = str(df["MATERNO"].iloc[0]).strip().upper() if "MATERNO" in df.columns and pd.notna(df["MATERNO"].iloc[0]) else ""
            
            # Reconstruct full input text
            full_input = " ".join([p for p in [gt_nom, gt_pat, gt_mat] if p])
            tokens = full_input.split()
            if not tokens:
                continue
            
            total += 1
            pred_tags = model_predict_fn(tokens)
            is_comp, p_nom, p_pat, p_mat = decode_bilou_prediction(tokens, pred_tags)
            
            # Check match
            match_nom = (p_nom == gt_nom)
            match_pat = (p_pat == gt_pat)
            match_mat = (p_mat == gt_mat)
            
            if match_nom and match_pat and match_mat:
                correct_entities += 1
            else:
                print(f"  [Mismatch en {csv_file}]:")
                print(f"    GT:   Nom='{gt_nom}' | Pat='{gt_pat}' | Mat='{gt_mat}'")
                print(f"    PRED: Nom='{p_nom}' | Pat='{p_pat}' | Mat='{p_mat}' (is_comp={is_comp})")
        except Exception as e:
            pass
            
    acc = (correct_entities / total * 100.0) if total > 0 else 0.0
    print(f"Resultado en Ground Truth Real: {correct_entities}/{total} ({acc:.2f}%) exactitud perfecta")
    return acc, total

def main():
    print("=== [OPCIÓN A] Entrenamiento y Evaluación de CRF (Conditional Random Fields) ===")
    
    print("1. Cargando datasets generados...")
    X_train_raw, y_train = load_dataset_file("train.jsonl")
    X_val_raw, y_val = load_dataset_file("val.jsonl")
    X_test_raw, y_test = load_dataset_file("test.jsonl")
    
    print(f"   Train: {len(X_train_raw)} | Val: {len(X_val_raw)} | Test: {len(X_test_raw)}")
    
    print("2. Extrayendo features lingüísticas y de diccionario...")
    t0 = time.time()
    X_train = [sent2features(s) for s in X_train_raw]
    X_test = [sent2features(s) for s in X_test_raw]
    feat_time = time.time() - t0
    print(f"   Features extraídas en {feat_time:.2f} s")
    
    print("3. Entrenando CRF con optimizador L-BFGS y regularización L1+L2...")
    crf = CRF(
        algorithm='lbfgs',
        c1=0.1, # L1 penalty for sparsity
        c2=0.1, # L2 penalty
        max_iterations=100,
        all_possible_transitions=True,
        verbose=False
    )
    
    t_train_start = time.time()
    crf.fit(X_train, y_train)
    train_duration = time.time() - t_train_start
    print(f"   Entrenamiento completado en {train_duration:.2f} segundos!")
    
    # Save model
    model_path = os.path.join(MODELS_DIR, "crf_ner_model.joblib")
    joblib.dump(crf, model_path)
    model_size_mb = os.path.getsize(model_path) / (1024 * 1024)
    print(f"   Modelo guardado en: {model_path} ({model_size_mb:.2f} MB)")
    
    # 4. Evaluation on synthetic test split
    print("\n4. Evaluando en conjunto de prueba sintético (6,000 secuencias)...")
    t_pred_start = time.time()
    y_pred = crf.predict(X_test)
    pred_duration = time.time() - t_pred_start
    latency_per_sample_us = (pred_duration / len(X_test)) * 1_000_000 # microseconds
    
    labels = list(crf.classes_)
    f1_micro = crf_metrics.flat_f1_score(y_test, y_pred, average='weighted', labels=labels)
    acc = crf_metrics.flat_accuracy_score(y_test, y_pred)
    print(f"   Exactitud por Token (Token Accuracy): {acc*100:.2f}%")
    print(f"   F1-Score Ponderado: {f1_micro*100:.2f}%")
    print(f"   Latencia promedio de inferencia: {latency_per_sample_us:.1f} microsegundos por nombre (~{1_000_000/latency_per_sample_us:.0f} nombres/segundo)")
    
    print("\nReporte de Clasificación por Etiqueta BILOU:")
    report = crf_metrics.flat_classification_report(y_test, y_pred, labels=labels, digits=4)
    print(report)
    
    # 5. Real-world evaluation
    def predict_tokens(tokens):
        feats = [word2features(tokens, i) for i in range(len(tokens))]
        return crf.predict([feats])[0]
        
    real_acc, real_total = evaluate_on_real_ground_truth(predict_tokens)
    
    # Save summary results
    results = {
        "model": "CRF (Conditional Random Fields)",
        "train_time_sec": train_duration,
        "token_accuracy": acc,
        "f1_score": f1_micro,
        "latency_us": latency_per_sample_us,
        "model_size_mb": model_size_mb,
        "real_gt_accuracy": real_acc
    }
    with open(os.path.join(LAB_DIR, "crf_benchmark_results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)
        
    print("\nBenchmark de CRF concluido.")

if __name__ == "__main__":
    main()
