import os
import time
import json
import numpy as np
import cv2
import fitz
import easyocr
from rapidocr_onnxruntime import RapidOCR

# ---------------------------------------------------------
# 1. Define Ground Truth for Page 1
# ---------------------------------------------------------
GROUND_TRUTH_TEXT = (
    "Este documento contiene información RESERVADA y CONFIDENCIAL de "
    "conformidad con lo establecido en el artículo 112, fracciones I, IV, VII y XVII; y "
    "115 de la Ley General de Transparencia y Acceso a la Información Pública, por "
    "lo que no deberá darse a conocer su contenido. "
    "Dirección General de Integración de la Lista de Personas Bloqueadas y Procedimientos de Garantías de Audiencia "
    "Oficio No. 110/G/1329/2026 "
    "Ciudad de México, a 26 de mayo de 2026 "
    "Mtro. Juan Ayax Fuentes Mendoza "
    "Vicepresidente de Supervisión de Procesos Preventivos, de la Comisión Nacional Bancaria y de Valores. "
    "P r e s e n t e "
    "Conforme a las atribuciones previstas en la fracción XXXIV, del artículo 10 y; V, del artículo 10-G, del "
    "Reglamento Interior de la Secretaría de Hacienda y Crédito Público, en relación con lo dispuesto en "
    "el párrafo decimonoveno del artículo 50 del mismo Reglamento, así como la 70ª y 71 ª de las "
    "Disposiciones de Carácter General a que se refiere el artículo 115 de la Ley de Instituciones de "
    "Crédito, el que suscribe, en suplencia del Titular de la Unidad de Inteligencia Financiera, comunico la "
    "actualización de la Lista de Personas Bloqueadas, mediante el Acuerdo 171/2026, emitido por el "
    "Titular de esta Unidad de Inteligencia Financiera. "
    "En ese supuesto, con fundamento en el artículo 95 Bis de la Ley General de Organizaciones y "
    "Actividades Auxiliares del Crédito; 61ª, 62ª, 63ª y 64ª, de las Disposiciones de Carácter General a que "
    "se refieren los artículos 115 de la Ley de Instituciones de Crédito en relación con el 87-D de la Ley "
    "General de Organizaciones y Actividades Auxiliares del Crédito y 95-Bis de este último ordenamiento, "
    "aplicables a las sociedades financieras de objeto múltiple; 57ª, 58ª, 59ª y 60ª, de las Disposiciones de "
    "Carácter General a que se refiere el artículo 95 Bis de la Ley General de Organizaciones y Actividades "
    "Auxiliares del Crédito aplicables a los centros cambiarios a que se refiere el artículo 81-A del mismo "
    "documento legal; y 60ª, 61ª, 62ª y 63ª, de las Disposiciones de Carácter General a que se refiere el "
    "artículo 95 Bis de la Ley General de Organizaciones y Actividades Auxiliares del Crédito, aplicables a "
    "los transmisores de dinero a que se refiere el artículo 81-A Bis del mismo precepto, las entidades "
    "financieras deberán proceder a dar atención a los siguientes puntos: "
    "1.- Identificar al cliente o usuario que se encuentre en la Lista de Personas "
    "Bloqueadas mediante Acuerdo 171/2026, emitido el 26 de mayo de 2026, por el "
    "Titular de la Unidad de Inteligencia Financiera: "
    "N.º NOMBRE FC "
    "1 OPERADORA Y DESARROLLADORA DE INDUSTRIAS PODEBI SJE SAPI DE CV 30/11/2023 "
    "2026 año de Margarita Maza Página 1 de 2 Av. Constituyentes 1001, Col. Belén de las Flores, Alc. Álvaro Obregón, Ciudad de México Tel: (55) 0000 0000. www.gob.mx/uif"
)

# ---------------------------------------------------------
# 2. Levenshtein Distance & Accuracy metrics
# ---------------------------------------------------------
def levenshtein_distance(s1, s2):
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    
    previous_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
        
    return previous_row[-1]

def clean_text(text):
    # Standardize whitespace and convert to lowercase
    return " ".join(text.lower().split())

def evaluate_accuracy(ocr_text, gt_text=GROUND_TRUTH_TEXT):
    ocr_clean = clean_text(ocr_text)
    gt_clean = clean_text(gt_text)
    
    # Calculate Character Error Rate (CER)
    dist_char = levenshtein_distance(ocr_clean, gt_clean)
    cer = (dist_char / len(gt_clean)) * 100 if len(gt_clean) > 0 else 0
    accuracy = 100 - cer
    
    # Calculate Word Error Rate (WER)
    ocr_words = ocr_clean.split()
    gt_words = gt_clean.split()
    dist_word = levenshtein_distance(ocr_words, gt_words)
    wer = (dist_word / len(gt_words)) * 100 if len(gt_words) > 0 else 0
    
    return {
        "cer": round(cer, 2),
        "wer": round(wer, 2),
        "accuracy": round(accuracy, 2),
        "char_dist": dist_char,
        "word_dist": dist_word,
        "extracted_chars": len(ocr_clean),
        "gt_chars": len(gt_clean)
    }

# ---------------------------------------------------------
# 3. Main Evaluation Pipeline
# ---------------------------------------------------------
def main():
    pdf_path = "files/lpb_oficios_01_06_26/8_operadora_y_desarrolladora_de_industrias.pdf"
    print(f"Opening PDF: {pdf_path}")
    doc = fitz.open(pdf_path)
    page = doc[0]  # page 1
    
    # Pre-initialize readers to avoid counting loading time in execution speed benchmarks
    print("Pre-initializing EasyOCR Reader...")
    easyocr_reader = easyocr.Reader(['es', 'en'], gpu=False)
    
    print("Pre-initializing RapidOCR Reader...")
    rapidocr_engine = RapidOCR()
    
    results = {}
    
    # Test cases: (Engine name, DPI, execute_func)
    test_cases = [
        {
            "name": "EasyOCR (DPI=150)",
            "dpi": 150,
            "engine": "easyocr"
        },
        {
            "name": "EasyOCR (DPI=300)",
            "dpi": 300,
            "engine": "easyocr"
        },
        {
            "name": "RapidOCR (DPI=150)",
            "dpi": 150,
            "engine": "rapidocr"
        },
        {
            "name": "RapidOCR (DPI=300)",
            "dpi": 300,
            "engine": "rapidocr"
        }
    ]
    
    for case in test_cases:
        name = case["name"]
        dpi = case["dpi"]
        engine_type = case["engine"]
        
        print(f"\nEvaluating: {name}...")
        
        # Render page
        pix = page.get_pixmap(dpi=dpi)
        img_data = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
        if pix.n == 4:
            img_data = cv2.cvtColor(img_data, cv2.COLOR_RGBA2RGB)
        elif pix.n == 1:
            img_data = cv2.cvtColor(img_data, cv2.COLOR_GRAY2RGB)
            
        start_time = time.time()
        
        extracted_text = ""
        if engine_type == "easyocr":
            # Run EasyOCR
            ocr_res = easyocr_reader.readtext(img_data)
            extracted_text = " ".join([item[1] for item in ocr_res])
        elif engine_type == "rapidocr":
            # Run RapidOCR
            ocr_res, _ = rapidocr_engine(img_data)
            if ocr_res:
                extracted_text = " ".join([item[1] for item in ocr_res])
                
        elapsed = time.time() - start_time
        metrics = evaluate_accuracy(extracted_text)
        
        results[name] = {
            "time_seconds": round(elapsed, 2),
            "cer_percent": metrics["cer"],
            "wer_percent": metrics["wer"],
            "accuracy_percent": metrics["accuracy"],
            "char_errors": metrics["char_dist"],
            "word_errors": metrics["word_dist"],
            "text_sample": extracted_text[:200] + "..."
        }
        
        print(f"  Execution Time: {results[name]['time_seconds']}s")
        print(f"  Character Accuracy: {results[name]['accuracy_percent']}% (CER: {results[name]['cer_percent']}%)")
        print(f"  Word Error Rate: {results[name]['wer_percent']}%")

    # Save to JSON
    with open("ocr_accuracy_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)
        
    print("\n\n==========================================")
    print("             EVALUATION REPORT            ")
    print("==========================================")
    print(f"{'Method':<20} | {'Time (s)':<8} | {'CER (%)':<8} | {'WER (%)':<8} | {'Accuracy (%)':<12}")
    print("-" * 65)
    for name, res in results.items():
        print(f"{name:<20} | {res['time_seconds']:<8.2f} | {res['cer_percent']:<8.2f} | {res['wer_percent']:<8.2f} | {res['accuracy_percent']:<12.2f}")
    print("==========================================")

if __name__ == "__main__":
    main()
