import os
import random
import re
import json

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Set seed for reproducibility
random.seed(42)

def load_file_lines(filename):
    path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip().upper() for line in f if line.strip()]

def tag_sequence(tokens, entity_type):
    """Assigns BILOU tags to a list of tokens representing an entity."""
    n = len(tokens)
    if n == 0:
        return []
    if n == 1:
        return [f"U-{entity_type}"]
    tags = [f"B-{entity_type}"]
    for _ in range(n - 2):
        tags.append(f"I-{entity_type}")
    tags.append(f"L-{entity_type}")
    return tags

def apply_ocr_noise(text, p=0.15):
    """Simulates realistic OCR degradation on tokens."""
    if random.random() > p:
        return text
    
    replacements = {
        'O': '0', '0': 'O',
        'I': '1', '1': 'I',
        'L': '1',
        'S': '5', '5': 'S',
        'B': '8', '8': 'B',
        'Z': '2', '2': 'Z'
    }
    chars = list(text)
    for i in range(len(chars)):
        c = chars[i]
        if c in replacements and random.random() < 0.25:
            chars[i] = replacements[c]
            break
    return "".join(chars)

def clean_token(tok):
    return re.sub(r'[^A-Z0-9\.\,\-\&]', '', tok)

def generate_person_sample(first_names, surnames, compound_surnames):
    """Generates a labeled person sequence with realistic Mexican naming patterns."""
    pattern = random.choices(
        ["first_pat_mat", "first_second_pat_mat", "compound_first_pat_mat", "first_pat_only", "compound_pat_mat"],
        weights=[0.35, 0.35, 0.15, 0.05, 0.10],
        k=1
    )[0]
    
    tokens = []
    labels = []
    
    if pattern == "first_pat_mat":
        fn = random.choice(first_names).split()
        pat = random.choice(surnames).split()
        mat = random.choice(surnames).split()
        
        tokens.extend(fn)
        labels.extend(tag_sequence(fn, "FIRST_NAME"))
        
        tokens.extend(pat)
        labels.extend(tag_sequence(pat, "PAT_SURNAME"))
        
        tokens.extend(mat)
        labels.extend(tag_sequence(mat, "MAT_SURNAME"))
        
    elif pattern == "first_second_pat_mat":
        fn = random.choice(first_names).split()[:1]
        sn = random.choice(first_names).split()[:1]
        pat = random.choice(surnames).split()
        mat = random.choice(surnames).split()
        
        tokens.extend(fn)
        labels.extend(tag_sequence(fn, "FIRST_NAME"))
        
        tokens.extend(sn)
        labels.extend(tag_sequence(sn, "SECOND_NAME"))
        
        tokens.extend(pat)
        labels.extend(tag_sequence(pat, "PAT_SURNAME"))
        
        tokens.extend(mat)
        labels.extend(tag_sequence(mat, "MAT_SURNAME"))
        
    elif pattern == "compound_first_pat_mat":
        # Compound first names like "MARIA DEL CARMEN", "JOSE DE JESUS", "ANA MARIA"
        compound_fn = random.choice([
            "MARIA DEL CARMEN", "MARIA DE LA LUZ", "MARIA GUADALUPE", "JOSE DE JESUS",
            "JOSE LUIS", "JUAN CARLOS", "ANA MARIA", "LUIS ALBERTO", "MIGUEL ANGEL",
            "BLANCA ESTELA", "ROSA MARIA", "VICTOR MANUEL", "CARLOS ALBERTO"
        ]).split()
        pat = random.choice(surnames).split()
        mat = random.choice(surnames).split()
        
        tokens.extend(compound_fn)
        labels.extend(tag_sequence(compound_fn, "FIRST_NAME"))
        
        tokens.extend(pat)
        labels.extend(tag_sequence(pat, "PAT_SURNAME"))
        
        tokens.extend(mat)
        labels.extend(tag_sequence(mat, "MAT_SURNAME"))
        
    elif pattern == "compound_pat_mat":
        fn = random.choice(first_names).split()[:2]
        # First name could be single or two tokens
        if len(fn) == 1:
            tokens.extend(fn)
            labels.extend(tag_sequence(fn, "FIRST_NAME"))
        else:
            tokens.append(fn[0])
            labels.extend(tag_sequence([fn[0]], "FIRST_NAME"))
            tokens.append(fn[1])
            labels.extend(tag_sequence([fn[1]], "SECOND_NAME"))
            
        pat = random.choice(compound_surnames).split()
        mat = random.choice(surnames).split()
        
        tokens.extend(pat)
        labels.extend(tag_sequence(pat, "PAT_SURNAME"))
        
        tokens.extend(mat)
        labels.extend(tag_sequence(mat, "MAT_SURNAME"))
        
    else: # first_pat_only (single surname / foreign)
        fn = random.choice(first_names).split()[:1]
        pat = random.choice(surnames).split()[:1]
        
        tokens.extend(fn)
        labels.extend(tag_sequence(fn, "FIRST_NAME"))
        
        tokens.extend(pat)
        labels.extend(tag_sequence(pat, "PAT_SURNAME"))
        
    # Apply noise to some tokens
    noisy_tokens = [apply_ocr_noise(t) for t in tokens]
    return noisy_tokens, labels

def generate_company_sample(companies):
    """Generates a labeled company sequence."""
    comp = random.choice(companies)
    # Split into words/tokens
    raw_tokens = [clean_token(t) for t in comp.split() if clean_token(t)]
    if not raw_tokens:
        raw_tokens = ["EMPRESA", "SA", "DE", "CV"]
    
    noisy_tokens = [apply_ocr_noise(t) for t in raw_tokens]
    labels = tag_sequence(noisy_tokens, "COMPANY")
    return noisy_tokens, labels

def main(total_samples=60000):
    print(f"=== Generando Dataset Sintético BILOU ({total_samples} muestras) ===")
    
    sat_companies = load_file_lines("sat_companies.txt")
    uif_companies = load_file_lines("uif_domain_companies.txt")
    all_companies = sat_companies + uif_companies * 20 # Upweight domain entities
    
    surnames = load_file_lines("sep_apellidos.txt")
    hombres = load_file_lines("sep_hombres.txt")
    mujeres = load_file_lines("sep_mujeres.txt")
    all_first_names = hombres + mujeres
    
    compound_surnames = [
        "DE LA ROSA", "DE LA GARZA", "DE LA PEÑA", "DE LA TORRE", "DE LA CRUZ",
        "DEL CASTILLO", "DEL TORO", "DEL VALLE", "DEL ANGEL", "DEL CARMEN",
        "DE ANDA", "DE LEON", "DE SANTIAGO", "DE LOS SANTOS", "DE DIOS",
        "MONTES DE OCA", "SAN MARTIN", "SAN JUAN", "SANCHEZ DE TAGLE"
    ]
    
    samples = []
    
    # 50% Personas Físicas, 50% Personas Morales (balance equilibrado)
    n_persons = total_samples // 2
    n_companies = total_samples - n_persons
    
    print(f"  Generando {n_persons} personas físicas...")
    for _ in range(n_persons):
        toks, labs = generate_person_sample(all_first_names, surnames, compound_surnames)
        if toks and len(toks) == len(labs):
            samples.append({"tokens": toks, "labels": labs, "type": "PERSON"})
            
    print(f"  Generando {n_companies} personas morales...")
    for _ in range(n_companies):
        toks, labs = generate_company_sample(all_companies)
        if toks and len(toks) == len(labs):
            samples.append({"tokens": toks, "labels": labs, "type": "COMPANY"})
            
    random.shuffle(samples)
    
    # Train / Val / Test split (80% / 10% / 10%)
    n_train = int(len(samples) * 0.80)
    n_val = int(len(samples) * 0.10)
    
    train_data = samples[:n_train]
    val_data = samples[n_train:n_train + n_val]
    test_data = samples[n_train + n_val:]
    
    print(f"\nDistribución del Dataset:")
    print(f"  - Train: {len(train_data)} secuencias")
    print(f"  - Val:   {len(val_data)} secuencias")
    print(f"  - Test:  {len(test_data)} secuencias")
    
    # Save splits as JSONL
    for name, data in [("train", train_data), ("val", val_data), ("test", test_data)]:
        path = os.path.join(OUTPUT_DIR, f"{name}.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"  Guardado {name}.jsonl en {path}")
        
    # Sample display
    print("\nEjemplo de muestra generada (Persona):")
    sample_person = next(s for s in train_data if s["type"] == "PERSON")
    for t, l in zip(sample_person["tokens"], sample_person["labels"]):
        print(f"  {t:<20} -> {l}")
        
    print("\nEjemplo de muestra generada (Empresa):")
    sample_comp = next(s for s in train_data if s["type"] == "COMPANY")
    for t, l in zip(sample_comp["tokens"], sample_comp["labels"]):
        print(f"  {t:<20} -> {l}")

if __name__ == "__main__":
    main()
