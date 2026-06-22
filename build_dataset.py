import os
import re
import csv
import pandas as pd
import fitz
import numpy as np
import cv2
from rapidocr_onnxruntime import RapidOCR

def clean_name(name):
    name = name.lower()
    name = re.sub(r"^(10_|11_|12_|13_|14_|15_|16_|17_|18_|19_|20_|21_|22_|23_|24_|25_|1_|2_|3_|4_|5_|6_|7_|8_|9_)", "", name)
    name = re.sub(r"(_cnbv|_vs|\.pdf|\.csv)", "", name)
    # Normalize accents / spanish chars to avoid minor differences
    name = name.replace("á", "a").replace("é", "e").replace("í", "i").replace("ó", "o").replace("ú", "u").replace("ñ", "n")
    # Remove all separator marks
    name = re.sub(r"[_\-\s]+", "", name)
    return name

def extract_office_number(text):
    # Regex to find patterns like 110/K/2924/2026 or 110-G-1329-2026
    m = re.search(r"110[/\-\s_]?[GKgk][/\-\s_]?\d+[/\-\s_]?\d+", text)
    if m:
        # Normalize to 110_k_2924_2026 format
        val = re.sub(r"[/\-\s_]+", "_", m.group(0).lower())
        return val
    return None

def main():
    os.makedirs("transcriptions", exist_ok=True)
    
    # 1. Initialize RapidOCR
    print("Initializing RapidOCR engine...")
    ocr_engine = RapidOCR()
    
    # 2. Gather all PDFs and CSVs
    pdf_files = []
    csv_files = []
    for root, dirs, files in os.walk("files"):
        for file in files:
            path = os.path.join(root, file)
            if file.lower().endswith(".pdf"):
                pdf_files.append((file, path, root))
            elif file.lower().endswith(".csv"):
                csv_files.append((file, path, root))
                
    print(f"Found {len(pdf_files)} PDFs and {len(csv_files)} CSVs.")
    
    # 3. Read and cache PDF transcriptions (Page 1 only)
    pdf_texts = {}
    for idx, (file, path, root) in enumerate(pdf_files):
        basename = os.path.splitext(file)[0]
        cache_path = os.path.join("transcriptions", f"{basename}.txt")
        
        # Check if already transcribed
        if os.path.exists(cache_path):
            with open(cache_path, "r", encoding="utf-8") as f:
                extracted_text = f.read()
        else:
            print(f"[{idx+1}/{len(pdf_files)}] Transcribing Page 1 of {file}...")
            try:
                doc = fitz.open(path)
                page = doc[0] # Page 1
                pix = page.get_pixmap(dpi=96)
                img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
                if pix.n == 4:
                    img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
                elif pix.n == 1:
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                
                res, _ = ocr_engine(img)
                extracted_text = "\n".join([item[1] for item in res]) if res else ""
                
                # Cache it
                with open(cache_path, "w", encoding="utf-8") as f:
                    f.write(extracted_text)
            except Exception as e:
                print(f"Error processing {file}: {e}")
                extracted_text = ""
                
        pdf_texts[path] = extracted_text
        
    # 4. Parse CSV metadata
    csv_records = []
    office_to_csv = {}
    
    for file, path, root in csv_files:
        try:
            with open(path, mode="r", encoding="utf-8", errors="ignore") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    movimiento = row.get("MOVIMIENTO", "").strip().upper()
                    nombre = row.get("NOMBRE", "").strip()
                    paterno = row.get("PATERNO", "").strip()
                    materno = row.get("MATERNO", "").strip()
                    motivo = row.get("MOTIVO", "").strip()
                    ciudad = row.get("CIUDAD", "").strip()
                    pais = row.get("PAIS", "").strip()
                    
                    record = {
                        "csv_file": file,
                        "csv_path": path,
                        "csv_dir": root,
                        "movimiento": movimiento,
                        "nombre": nombre,
                        "paterno": paterno,
                        "materno": materno,
                        "motivo": motivo,
                        "ciudad": ciudad,
                        "pais": pais
                    }
                    csv_records.append(record)
                    
                    # Extract normalized office number to map via office number
                    office_norm = None
                    for val in row.values():
                        if val:
                            office_norm = extract_office_number(str(val))
                            if office_norm:
                                break
                    if office_norm:
                        office_to_csv[office_norm] = record
        except Exception as e:
            print(f"Error reading CSV {file}: {e}")
            
    # 5. Perform alignment between PDFs and CSVs
    dataset = []
    unmatched_pdfs = []
    
    for file, path, root in pdf_files:
        text = pdf_texts.get(path, "")
        pdf_clean = clean_name(file)
        
        matched_record = None
        match_type = ""
        
        # Method 1: Substring name match in the same directory
        for record in csv_records:
            if record["csv_dir"] == root:
                csv_clean = clean_name(record["csv_file"])
                if csv_clean in pdf_clean or pdf_clean in csv_clean:
                    matched_record = record
                    match_type = "filename_match"
                    break
                    
        # Method 2: Office number from PDF filename
        if not matched_record:
            office_from_filename = extract_office_number(file)
            if office_from_filename and office_from_filename in office_to_csv:
                matched_record = office_to_csv[office_from_filename]
                match_type = "office_filename_match"
                
        # Method 3: Office number from PDF OCR text
        if not matched_record:
            office_from_text = extract_office_number(text)
            if office_from_text and office_from_text in office_to_csv:
                matched_record = office_to_csv[office_from_text]
                match_type = "office_text_match"
                
        if matched_record:
            # Build dataset row
            row = {
                "pdf_name": file,
                "pdf_path": path,
                "pdf_dir": os.path.basename(root),
                "extracted_text": text,
                "match_type": match_type,
                "csv_file": matched_record["csv_file"],
                "csv_path": matched_record["csv_path"],
                "label_movement": matched_record["movimiento"],
                "label_nombre": matched_record["nombre"],
                "label_paterno": matched_record["paterno"],
                "label_materno": matched_record["materno"],
                "label_motivo": matched_record["motivo"],
                "label_ciudad": matched_record["ciudad"],
                "label_pais": matched_record["pais"]
            }
            dataset.append(row)
        else:
            unmatched_pdfs.append((file, path))
            
    print(f"\nAlignment Complete:")
    print(f"Total Matched PDFs: {len(dataset)}")
    print(f"Total Unmatched PDFs: {len(unmatched_pdfs)}")
    
    # Save dataset to Parquet
    df = pd.DataFrame(dataset)
    df.to_parquet("files_dataset.parquet", index=False)
    print("Dataset saved to files_dataset.parquet successfully!")
    
    if unmatched_pdfs:
        print("\nList of unmatched PDFs (they lack matching CSV rows or office numbers):")
        for u_file, u_path in unmatched_pdfs[:10]:
            print(f"  - {u_file} ({u_path})")
        if len(unmatched_pdfs) > 10:
            print(f"  ... and {len(unmatched_pdfs) - 10} more.")

if __name__ == "__main__":
    main()
