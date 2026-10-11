"""
scripts/evaluate_outputs.py
============================
Compara los CSV que genera la app con los CSV de referencia.

Uso:
    python scripts/evaluate_outputs.py <carpeta_referencia> <carpeta_salida> <carpeta_pdfs> [-v]

<carpeta_referencia> y <carpeta_pdfs> deben tener las mismas subcarpetas por fecha
(p. ej. files/lpb_oficios_01_06_26). Cada PDF se empareja con el CSV de referencia
de su misma subcarpeta cuyo nombre más se parece al nombre del archivo o, si el
archivo no lleva nombre (110_k_*, escaneos mx_m5071_*), con el que tiene el mismo
número de oficio. Los PDF sin pareja clara no se cuentan.

Importante: la carpeta de PDFs que se procesa no debe contener los CSV de
referencia, para medir la app sin ayudas.
"""
import csv, re, sys, unicodedata
from pathlib import Path

def read_rows(p):
    raw = Path(p).read_bytes()
    for enc in ("utf-8-sig", "cp1252"):
        try:
            txt = raw.decode(enc); break
        except UnicodeDecodeError:
            continue
    return list(csv.DictReader(txt.splitlines()))

def fold(s):
    s = s.upper().replace("Ñ", "\0")
    s = "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))
    return s.replace("\0", "Ñ")

def key(s):
    return re.sub(r"[^A-Z]", "", fold(s).replace("Ñ", "N"))

def bigr(a, b):
    A = {a[i:i+2] for i in range(len(a)-1)}; B = {b[i:i+2] for i in range(len(b)-1)}
    return len(A & B) / len(A | B) if A | B else 0

ref_root, out_dir, pdf_root = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
FIELDS = ["MOVIMIENTO", "NOMBRE", "PATERNO", "MATERNO", "MOTIVO"]
ok = {f: 0 for f in FIELDS}; allok = 0; n = 0; enye_total = enye_ok = 0; misses = []
for date_dir in sorted(pdf_root.iterdir()):
    refs = []
    for c in (ref_root / date_dir.name).glob("*.csv"):
        for r in read_rows(c):
            full = " ".join(r.get(k, "") for k in ("NOMBRE", "PATERNO", "MATERNO"))
            refs.append((key(full), r))
    for pdf in sorted(date_dir.glob("*.pdf")):
        out_csv = out_dir / f"{pdf.stem}.csv"
        if not out_csv.exists():
            out_csv = next(out_dir.glob(f"{pdf.stem}*.csv"), out_csv)
        got = read_rows(out_csv)[0] if out_csv.exists() else None
        # 1) por nombre de archivo; 2) si no, por número de oficio (único en la carpeta)
        k = key(re.sub(r"^\d+_|_cnbv$", "", pdf.stem))
        best = max(refs, key=lambda x: bigr(k, x[0]), default=None)
        ref = best[1] if best and bigr(k, best[0]) >= 0.6 else None
        if ref is None and got and got.get("MOTIVO", "N/A") != "N/A":
            same = [r for _, r in refs if r.get("MOTIVO", "").strip().upper() == got["MOTIVO"].strip().upper()]
            ref = same[0] if len(same) == 1 else None
        if ref is None:
            continue
        if got is None:
            misses.append((pdf.name, "SIN SALIDA")); n += 1; continue
        n += 1; row_ok = True
        for f in FIELDS:
            if got.get(f, "").strip().upper() == ref.get(f, "").strip().upper():
                ok[f] += 1
            else:
                row_ok = False
        allok += row_ok
        full_ref = " ".join(ref.get(x, "") for x in ("NOMBRE", "PATERNO", "MATERNO"))
        if "Ñ" in full_ref.upper():
            enye_total += 1
            enye_ok += " ".join(got.get(x, "") for x in ("NOMBRE", "PATERNO", "MATERNO")).upper() == full_ref.upper()
        if not row_ok:
            misses.append((pdf.name, "|".join(got.get(f, "") for f in FIELDS[:4]), "|".join(ref.get(f, "") for f in FIELDS[:4])))
print(f"PDFs evaluados: {n}")
for f in FIELDS:
    print(f"  {f:<11} {ok[f]}/{n} ({100*ok[f]/n:.1f}%)")
print(f"  FILA COMPLETA {allok}/{n} ({100*allok/n:.1f}%)")
print(f"  Nombres con Ñ correctos: {enye_ok}/{enye_total}")
if "-v" in sys.argv:
    for m in misses: print("   ", *m)
