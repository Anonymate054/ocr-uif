import os
import io
import re
import requests
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
os.makedirs(DATA_DIR, exist_ok=True)

HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

def download_sat_companies():
    print("=== [1/3] Descargando y procesando listas oficiales del SAT (Art. 69-B) ===")
    sat_urls = [
        ("Definitivos", "http://omawww.sat.gob.mx/cifras_sat/Documents/Definitivos.csv"),
        ("Presuntos", "http://omawww.sat.gob.mx/cifras_sat/Documents/Presuntos.csv"),
        ("Desvirtuados", "http://omawww.sat.gob.mx/cifras_sat/Documents/Desvirtuados.csv"),
        ("SentenciasFavorables", "http://omawww.sat.gob.mx/cifras_sat/Documents/SentenciasFavorables.csv")
    ]
    
    companies = set()
    physicals_from_sat = []
    
    for name, url in sat_urls:
        try:
            print(f"  Descargando SAT {name}...")
            r = requests.get(url, headers=HEADERS, timeout=20)
            if r.status_code == 200:
                # SAT CSVs have 2 header rows
                df = pd.read_csv(io.BytesIO(r.content), encoding="latin1", skiprows=2, on_bad_lines="skip")
                
                # Identify RFC and Name columns
                rfc_col = None
                name_col = None
                for c in df.columns:
                    c_clean = str(c).strip().upper()
                    if "RFC" in c_clean:
                        rfc_col = c
                    elif "NOMBRE" in c_clean or "RAZ" in c_clean:
                        name_col = c
                
                if rfc_col and name_col:
                    for _, row in df.iterrows():
                        rfc = str(row[rfc_col]).strip().upper()
                        contrib = str(row[name_col]).strip().upper()
                        if not contrib or contrib == "NAN" or len(contrib) < 3:
                            continue
                        
                        # Clean unwanted quotes or double spaces
                        contrib = re.sub(r'[\"\']', '', contrib)
                        contrib = re.sub(r'\s+', ' ', contrib).strip()
                        
                        # In Mexican RFC: 12 chars = Persona Moral (Empresa), 13 chars = Persona Física
                        clean_rfc = re.sub(r'[^A-Z0-9]', '', rfc)
                        if len(clean_rfc) == 12:
                            companies.add(contrib)
                        elif len(clean_rfc) == 13:
                            physicals_from_sat.append(contrib)
                            
            print(f"    Subtotal empresas acumuladas: {len(companies)}")
        except Exception as e:
            print(f"  Aviso: error descargando {name}: {e}")
            
    out_companies = os.path.join(DATA_DIR, "sat_companies.txt")
    with open(out_companies, "w", encoding="utf-8") as f:
        for c in sorted(companies):
            f.write(c + "\n")
    print(f"-> Guardadas {len(companies)} empresas reales de México en: {out_companies}")
    return list(companies)

def download_sep_names():
    print("\n=== [2/3] Descargando catálogo oficial de Nombres y Apellidos de México (SEP/INEGI) ===")
    sep_base = "https://raw.githubusercontent.com/eduardofv/mexican-names/master/data/sep/"
    
    files = {
        "apellidos": "apellidos.csv",
        "hombres": "hombres.csv",
        "mujeres": "mujeres.csv"
    }
    
    results = {}
    for key, filename in files.items():
        url = sep_base + filename
        print(f"  Descargando {filename}...")
        r = requests.get(url, headers=HEADERS, timeout=15)
        if r.status_code == 200:
            df = pd.read_csv(io.BytesIO(r.content), encoding="latin1")
            col = "apellido" if key == "apellidos" else "nombre"
            items = df[col].dropna().astype(str).str.strip().str.upper().tolist()
            # Filter single letters or noise
            items = [it for it in items if len(it) > 1 and it.replace(" ", "").isalpha()]
            results[key] = items
            out_file = os.path.join(DATA_DIR, f"sep_{key}.txt")
            with open(out_file, "w", encoding="utf-8") as f:
                for it in items:
                    f.write(it + "\n")
            print(f"    Guardados {len(items)} registros en: {out_file}")
            
    return results

def add_domain_custom_vocabulary():
    print("\n=== [3/3] Incorporando vocabulario específico de oficios UIF y entidades financieras ===")
    # Legal terms, financial institutions, and common government-listed entities
    extra_companies = [
        "SIMETRIA JURIDICA SC", "MEC COMPETITIVIDAD AMERICANA SA DE CV", "INTEGRADORA EMPRESARIAL APE",
        "NUTRI Y RELAXATION SA DE CV", "SINDICATO PROGRESISTA DE GRASAS Y ACEITES",
        "CLINICA FISIOFIT HEALTH AND SPORT SA DE CV", "MKL 2022 SC", "PROGRAMAS DE PRODUCTIVIDAD STRATEGO SA DE CV",
        "EVERMORE BUSINESS CORPORATION SC", "JURIS CORPORACION LABORAL SC", "ASOCIACION DE PRODUCTIVIDAD Y COMPETITIVIDAD DE SONORA AC",
        "SISJUR INNOVACION CORPORATIVA SC", "ASOCIACION PATRONAL REGION ZAMORA AC", "CONFEDERACION DE SERVIDORES PUBLICOS DE LOS PODERES DE LOS ESTADOS",
        "OPERADORA Y DESARROLLADORA DE INDUSTRIAS SA DE CV", "ASOCIACION PATRONAL VANGUARDISTA AC",
        "ALIANZA CORPORATIVA CAMARENCE AC", "ASFALT PRO DESARROLLOS Y EDIFICACIONES SA DE CV",
        "ASOCIACION DE TRABAJADORES A LA VANGUARDIA AC", "BMF ACCESORIOS Y NOVEDADES SA DE CV",
        "CONSULTORIA Y DESARROLLO INTEGRAL ADMINISTRATIVO SA DE CV", "CORPORATIVO JURIDICO DE PLANEACION LEGAL SC",
        "DESARROLLO 1540 SA DE CV", "GRUPO CONSTRUCTOR GASPEC SAS DE CV", "GLOBAL MINING LOGISTICS SA DE CV"
    ]
    out_extra = os.path.join(DATA_DIR, "uif_domain_companies.txt")
    with open(out_extra, "w", encoding="utf-8") as f:
        for c in extra_companies:
            f.write(c.upper() + "\n")
    print(f"-> Guardadas {len(extra_companies)} empresas de referencia de dominio UIF en: {out_extra}")

if __name__ == "__main__":
    download_sat_companies()
    download_sep_names()
    add_domain_custom_vocabulary()
    print("\n¡Descarga y preparación de datos públicos completada exitosamente!")
