import os
import re
import pandas as pd
import joblib

def get_char_bigrams(s):
    return set(s[i:i+2] for i in range(len(s)-1))

def bigram_similarity(s1, s2):
    b1 = get_char_bigrams(s1)
    b2 = get_char_bigrams(s2)
    if not b1 and not b2:
        return 1.0
    return len(b1.intersection(b2)) / len(b1.union(b2))

def extract_office_number(text):
    # Regex to find patterns like 110/K/2924/2026 or 110-G-1329-2026
    m = re.search(r"110[/\-\s_]?[GKgk][/\-\s_]?\d+[/\-\s_]?\d+", text)
    if m:
        # Normalize to standard 110/K/2924/2026 format
        parts = re.split(r"[/\-\s_]+", m.group(0).upper())
        if len(parts) >= 4:
            return f"OFICIO NO. {parts[0]}/{parts[1]}/{parts[2]}/{parts[3]}"
    return "N/A"

def extract_name(text):
    # Normalize spacing and line endings
    text_clean = re.sub(r"[ \t]+", " ", text)
    lines = [line.strip() for line in text_clean.split("\n") if line.strip()]
    
    # Reject list for generic legal terms
    rejects = ["DICHA PERSONA", "DICHAS PERSONAS", "LA QUEJOSA", "EL QUEJOSO", "LAS QUEJOSAS", "LOS QUEJOSOS", "EL ELIMINADO", "LOS ELIMINADOS", "LA SOCIEDAD", "LAS SOCIEDADES"]
    
    # Heuristic to check if a name is actually a sentence or legal term
    def is_invalid_name(name):
        name_upper = name.upper()
        invalid_keywords = [
            "LISTA DE PERSONAS", "LISTA", "BLOQUEADAS", "CONGELACION", "ACTIVOS", "SANCIONES", 
            "RESOLUCION", "ACUERDO", "OFICIO", "ELIMINACION", "CONGELAR", "BLOQUEAR",
            "ESTADO ISLAMICO", "CONSEJO DE SEGURIDAD", "NACIONES UNIDAS", "JUICIO DE AMPARO",
            "TRIBUNAL", "COLEGIADO", "JUZGADO", "DISTRITO", "REVISION", "PROHIBICION", "EMBARGO",
            "DE VIAJAR", "ARMAS", "N.O", "MATERIA", "HACIENDA", "SECRETARIA"
        ]
        for kw in invalid_keywords:
            if kw in name_upper:
                return True
        return False

    # Common Spanish first names that get merged with the second name
    def clean_ocr_name(name):
        prefixes = ["JOSE", "MARIA", "JUAN", "LUIS", "ANA", "SAN"]
        words = name.split()
        cleaned_words = []
        for w in words:
            w_upper = w.upper()
            split_done = False
            for prefix in prefixes:
                if w_upper.startswith(prefix) and len(w_upper) > len(prefix):
                    rest = w[len(prefix):]
                    if rest.isalpha():
                        cleaned_words.append(w[:len(prefix)])
                        cleaned_words.append(rest)
                        split_done = True
                        break
            if not split_done:
                cleaned_words.append(w)
        return " ".join(cleaned_words)

    # 1. Try BAJA / Se elimina / Resolution sentences first (very precise regex patterns)
    patterns = [
        # Se elimina de la Lista de Personas Bloqueadas a: <NAME>
        r"elimina,?\s*(?:de\s+la\s+Lista\s+de\s+Personas\s+Bloqueadas\s+)?a\s*:\s*(?:\d\)\s*)?([a-zA-Z\s\n\.,_-]{3,120}?)(?:\n\n|\r|\n|;|,|\s*con\s*R\.?F\.?C\.?|\.)",
        # eliminacion de <NAME>, con RFC
        r"eliminaci[oó6]n,?\s*de\s*([a-zA-Z\s\n\.,_-]{3,120}?)(?:,|\s*con\s*R\.?F\.?C\.?|;|\s*de\s*la\s*Lista|\n\n)",
        # impuesta a <NAME>
        r"impuesta,?\s*a\s*([a-zA-Z\s\n\.,_-]{3,120}?)(?:,|\s*con\s*R\.?F\.?C\.?|;|\s*de\s*la\s*Lista|\.|\n\n)",
        # impuestaa <NAME>
        r"impuestaa,?\s*([a-zA-Z\s\n\.,_-]{3,120}?)(?:,|\s*con\s*R\.?F\.?C\.?|;|\s*de\s*la\s*Lista|\.|\n\n)",
        # hace a <NAME>
        r"hace,?\s*a\s*([a-zA-Z\s\n\.,_-]{3,120}?)(?:,|\s*con\s*R\.?F\.?C\.?|;|\s*de\s*la\s*Lista|\.|\n\n)",
        # sanciones a: 1) <NAME>
        r"sanciones,?\s*a\s*:\s*(?:\d\)\s*)?([a-zA-Z\s\n\.,_-]{3,120}?)(?:;|,|\s*con\s*R\.?F\.?C\.?|\.|\n\n)",
        # suspendidos a <NAME>
        r"suspendidos,?\s*a\s*([a-zA-Z\s\n\.,_-]{3,120}?)(?:,|\s*con\s*R\.?F\.?C\.?|;|\s*de\s*la\s*Lista|\.|\n\n)",
        # suspendidosa <NAME>
        r"suspendidosa,?\s*([a-zA-Z\s\n\.,_-]{3,120}?)(?:,|\s*con\s*R\.?F\.?C\.?|;|\s*de\s*la\s*Lista|\.|\n\n)",
        # amparo: quejosa <NAME>
        r"quejosa\s+([a-zA-Z\s\n\.,_-]{3,120}?)(?:\.|\n\n|con\s+R\.?F\.?C\.?)"
    ]
    
    for pat in patterns:
        for m in re.finditer(pat, text_clean, re.IGNORECASE):
            name_cand = m.group(1).strip()
            name_cand = re.sub(r"\s+", " ", name_cand)
            name_cand = re.sub(r"[,;\.\s]+(?:con|de|la|R\.?F\.?C\.?|C\.?O\.?N\.?|con\s+R\.?F\.?C\.?)*$", "", name_cand, flags=re.IGNORECASE).strip()
            name_cand = re.sub(r"^[,\s\.]+", "", name_cand).strip()
            if len(name_cand.split()) >= 1:
                name_upper = name_cand.upper()
                if name_upper not in rejects and not is_invalid_name(name_upper):
                    return clean_ocr_name(name_upper)
                
    # 2. Try table-based extraction (often in ALTA documents)
    for i, line in enumerate(lines):
        if line.upper() in ["NOMBRE", "N.O NOMBRE", "N.O", "NOMBRE RFC", "FC NOMBRE", "PERSONA"]:
            name_lines = []
            for j in range(i + 1, min(i + 10, len(lines))):
                candidate = lines[j]
                # Skip headers and metadata tags
                if candidate.upper() in ["RFC", "FC", "N.O", "CURP", "FECHA", "NOMBRE", "PERSONA", "N.O."]:
                    continue
                # Stop conditions: date, pure numbers, or footer
                if re.search(r"\d{2}/\d{2}/\d{4}", candidate):
                    break
                if re.search(r"^[0-9]+$", candidate):
                    if int(candidate) < 100:
                        continue # Skip row index
                    else:
                        break
                if "av. constituyentes" in candidate.lower() or "margarita maza" in candidate.lower() or "pagina" in candidate.lower():
                    break
                # If candidate looks like RFC, stop/skip
                if len(candidate.split()) == 1 and any(char.isdigit() for char in candidate) and len(candidate) > 8:
                    break
                # Check uppercase letter ratio
                letters = re.sub(r"[^A-Za-z]+", "", candidate)
                if letters and sum(1 for c in letters if c.isupper()) > len(letters) * 0.5:
                    name_lines.append(candidate)
            if name_lines:
                # Filter out single-word candidates containing numbers
                filtered = [l for l in name_lines if not (len(l.split()) == 1 and any(c.isdigit() for c in l))]
                if filtered:
                    res_name = " ".join(filtered).upper()
                    if res_name not in rejects and not is_invalid_name(res_name):
                        return clean_ocr_name(res_name)
                    
    # 3. Fallback: find the first line in the text that is long, fully uppercase, and has no numbers
    skip_keywords = ["HACIENDA", "INTELI", "FINAN", "UNIDAD", "MEXICO", "RESOLU", "PRESENTE", "OFICIO", "DIREC", "GENERAL", "LISTA", "PERSONA", "BLOQ", "PROCED", "GARANT", "AUDIENCIA", "PAGINA"]
    for line in lines:
        if len(line) > 10 and line.isupper() and not any(c.isdigit() for c in line):
            # Skip common headers
            if any(k in line.upper() for k in skip_keywords):
                continue
            res_name = line.upper()
            if res_name not in rejects and not is_invalid_name(res_name):
                return clean_ocr_name(res_name)
            
    return "N/A"

def split_full_name(fullname, is_company):
    if not fullname or fullname == "N/A":
        return "", "", ""
    
    if is_company:
        return fullname, "", ""
    
    # Split person name (Nombre, Paterno, Materno)
    parts = fullname.split()
    if len(parts) == 0:
        return "", "", ""
    elif len(parts) == 1:
        return parts[0], "", ""
    elif len(parts) == 2:
        return parts[0], parts[1], ""
    elif len(parts) == 3:
        return parts[0], parts[1], parts[2]
    else:
        # 4 or more parts (e.g. JUAN ANTONIO ALISEDA ALCANTARA)
        # Assume first two are Nombre, third is Paterno, fourth is Materno
        return " ".join(parts[:-2]), parts[-2], parts[-1]

def is_moral_entity(name):
    # Heuristics to check if it is a company
    company_keywords = [
        "SA", "CV", "SC", "AC", "SAPI", "SOFOM", "SDR", "RL", "SOFIPO", "SNC", "GROUP", "GLOBAL",
        "INDUSTRIAS", "ASOCIACION", "SINDICATO", "CORPORACION", "CLINICA", "PROYECTOS", "DESARROLLOS",
        "EDIFICACIONES", "INNOVACION", "PATRONAL", "CONFEDERACION", "JURIDICA", "JURIDICO", "SERVICIOS",
        "LOGISTICA", "MEDICINA", "TRANSPORTES", "CONSTRUCTORA", "PROGRESISTA", "COMPETITIVIDAD",
        "COOPERATIVA", "BIENES", "INVERSIONES", "FINANCIERA", "ESTUDIOS", "DISTRIBUIDORA", "PRODUCTIVIDAD"
    ]
    name_upper = name.upper()
    words = re.split(r"[_\-\s\.]+", name_upper)
    for w in words:
        if w in company_keywords:
            return True
    return False

def main():
    # Load dataset
    df = pd.read_parquet("files_dataset.parquet")
    
    # Load champion SVM model and vectorizer
    vectorizer = joblib.load("models/tfidf_vectorizer.joblib")
    clf = joblib.load("models/svm_classifier_model.joblib")
    
    print(f"Loaded {len(df)} matched files from files_dataset.parquet.")
    
    matched_records = []
    name_correct_count = 0
    motivo_correct_count = 0
    
    for idx, row in df.iterrows():
        text = row["extracted_text"]
        pdf_name = row["pdf_name"]
        
        # Predict movement using champion SVM
        X_vec = vectorizer.transform([text])
        pred_movement = clf.predict(X_vec)[0]
        
        # Extract metadata from text
        extracted_oficio = extract_office_number(text)
        extracted_fullname = extract_name(text)
        
        # Reconstruct full ground-truth name
        gt_parts = [row["label_nombre"], row["label_paterno"], row["label_materno"]]
        gt_fullname = " ".join([str(p).strip() for p in gt_parts if pd.notna(p) and str(p).strip()])
        
        gt_motivo = row["label_motivo"]
        
        # Clean both for comparison
        def clean_str(s):
            return re.sub(r"[^A-Z0-9]+", "", str(s).upper())
        
        c_ext = clean_str(extracted_fullname)
        c_gt = clean_str(gt_fullname)
        
        # Fuzzy bigram similarity for OCR typo robustness
        sim = bigram_similarity(c_ext, c_gt)
        
        # Check match (exact, substring, or fuzzy similarity >= 0.60)
        is_name_correct = (c_ext == c_gt) or (len(c_gt) > 4 and c_gt in c_ext) or (len(c_ext) > 4 and c_ext in c_gt) or (sim >= 0.60)
        is_motivo_correct = clean_str(extracted_oficio) == clean_str(gt_motivo)
        
        if is_name_correct:
            name_correct_count += 1
        if is_motivo_correct:
            motivo_correct_count += 1
            
        is_company = is_moral_entity(extracted_fullname)
        nombre, paterno, materno = split_full_name(extracted_fullname, is_company)
        
        matched_records.append({
            "PDF Name": pdf_name,
            "GT Movement": row["label_movement"],
            "Pred Movement": pred_movement,
            "GT Name": gt_fullname,
            "Extracted Name": extracted_fullname,
            "Is Name Match": is_name_correct,
            "GT Oficio": gt_motivo,
            "Extracted Oficio": extracted_oficio,
            "Is Oficio Match": is_motivo_correct,
            "Is Company": is_company,
            "Nombre": nombre,
            "Paterno": paterno,
            "Materno": materno
        })
        
    df_results = pd.DataFrame(matched_records)
    print("\n=== METADATA EXTRACTION ACCURACY ===")
    print(f"Office Number Match Rate: {motivo_correct_count}/{len(df)} ({motivo_correct_count/len(df)*100:.2f}%)")
    print(f"Name Match Rate (Fuzzy/Cleaned): {name_correct_count}/{len(df)} ({name_correct_count/len(df)*100:.2f}%)")
    
    # Save a sample comparison CSV
    df_results.to_csv("metadata_extraction_validation.csv", index=False)
    print("\nSaved evaluation results to metadata_extraction_validation.csv.")
    
    # Show some mismatches to help refine rules
    mismatches = df_results[df_results["Is Name Match"] == False].head(10)
    if len(mismatches) > 0:
        print("\nFirst 10 Name Extraction Mismatches:")
        for idx, r in mismatches.iterrows():
            print(f"  - PDF: {r['PDF Name']}")
            print(f"    GT Name:        {r['GT Name']}")
            print(f"    Extracted Name: {r['Extracted Name']}")
            
if __name__ == "__main__":
    main()
