import os
import json
import time
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForTokenClassification, get_linear_schedule_with_warmup
import onnxruntime as ort
import numpy as np
import pandas as pd

LAB_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(LAB_DIR, "dataset")
MODELS_DIR = os.path.join(LAB_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
BATCH_SIZE = 32
EPOCHS = 2
LR = 3e-5
MAX_LEN = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_jsonl(filename):
    path = os.path.join(DATASET_DIR, filename)
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data

def build_label_vocab(all_data):
    unique_labels = set()
    for item in all_data:
        for l in item["labels"]:
            unique_labels.add(l)
    label2id = {l: i for i, l in enumerate(sorted(unique_labels))}
    id2label = {i: l for l, i in label2id.items()}
    return label2id, id2label

class NERDataset(Dataset):
    def __init__(self, data, tokenizer, label2id, max_len=MAX_LEN):
        self.data = data
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        tokens = item["tokens"]
        labels = item["labels"]

        encoding = self.tokenizer(
            tokens,
            is_split_into_words=True,
            return_offsets_mapping=False,
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt"
        )

        word_ids = encoding.word_ids(batch_index=0)
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx < len(labels):
                label_ids.append(self.label2id[labels[word_idx]])
            else:
                label_ids.append(-100)

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(label_ids, dtype=torch.long)
        }

def evaluate_on_real_ground_truth(predict_fn):
    test_dirs = [
        r"C:\Users\Anony\Documents\projects\OCR-UIF\test",
        r"C:\Users\Anony\Documents\GitHub\ocr-uif\test"
    ]
    test_dir = next((d for d in test_dirs if os.path.exists(d)), None)
    if not test_dir:
        return 0.0, 0
    
    csv_files = [f for f in os.listdir(test_dir) if f.endswith(".csv")]
    correct = 0
    total = 0
    
    for csv_file in csv_files:
        path = os.path.join(test_dir, csv_file)
        try:
            df = pd.read_csv(path)
            if df.empty or "NOMBRE" not in df.columns:
                continue
            gt_nom = str(df["NOMBRE"].iloc[0]).strip().upper() if pd.notna(df["NOMBRE"].iloc[0]) else ""
            gt_pat = str(df["PATERNO"].iloc[0]).strip().upper() if "PATERNO" in df.columns and pd.notna(df["PATERNO"].iloc[0]) else ""
            gt_mat = str(df["MATERNO"].iloc[0]).strip().upper() if "MATERNO" in df.columns and pd.notna(df["MATERNO"].iloc[0]) else ""
            
            full_input = " ".join([p for p in [gt_nom, gt_pat, gt_mat] if p])
            tokens = full_input.split()
            if not tokens:
                continue
            
            total += 1
            is_comp, p_nom, p_pat, p_mat = predict_fn(tokens)
            if p_nom == gt_nom and p_pat == gt_pat and p_mat == gt_mat:
                correct += 1
        except Exception:
            pass
            
    acc = (correct / total * 100.0) if total > 0 else 0.0
    return acc, total

def main():
    print(f"=== [OPCIÓN B] Entrenamiento Transformer (MiniLM-L12) en {DEVICE} ===")
    print(f"Cargando tokenizer: {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    train_data = load_jsonl("train.jsonl")[:16000] # 16k samples for fast convergence
    val_data = load_jsonl("val.jsonl")[:3000]
    test_data = load_jsonl("test.jsonl")[:3000]
    
    label2id, id2label = build_label_vocab(train_data)
    print(f"Clases detectadas ({len(label2id)} etiquetas): {list(label2id.keys())}")
    
    # Save label mappings for ONNX runtime inference
    labels_config = {"label2id": label2id, "id2label": id2label}
    with open(os.path.join(MODELS_DIR, "transformer_labels.json"), "w", encoding="utf-8") as f:
        json.dump(labels_config, f, indent=4)
        
    train_ds = NERDataset(train_data, tokenizer, label2id)
    val_ds = NERDataset(val_data, tokenizer, label2id)
    test_ds = NERDataset(test_data, tokenizer, label2id)
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)
    
    print(f"Inicializando modelo con {len(label2id)} etiquetas en {DEVICE}...")
    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id
    ).to(DEVICE)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(total_steps*0.1), num_training_steps=total_steps)
    
    t_start = time.time()
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for step, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
            
            if (step + 1) % 250 == 0:
                print(f"  Epoch {epoch+1}/{EPOCHS} | Paso {step+1}/{len(train_loader)} | Loss: {total_loss/(step+1):.4f}")
                
    train_duration = time.time() - t_start
    print(f"\nEntrenamiento Transformer completado en {train_duration:.2f} s en {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    
    # Save PyTorch checkpoint
    pt_path = os.path.join(MODELS_DIR, "minilm_pytorch.pt")
    torch.save(model.state_dict(), pt_path)
    print(f"Pesos PyTorch guardados en: {pt_path}")
    
    # Export to ONNX
    print("\n--- Exportando Transformer a formato ONNX ---")
    model.eval().to("cpu")
    onnx_path = os.path.join(MODELS_DIR, "minilm_ner.onnx")
    dummy_input = torch.ones(1, MAX_LEN, dtype=torch.long)
    dummy_mask = torch.ones(1, MAX_LEN, dtype=torch.long)
    
    torch.onnx.export(
        model,
        (dummy_input, dummy_mask),
        onnx_path,
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={"input_ids": {0: "batch_size", 1: "seq_len"}, "attention_mask": {0: "batch_size", 1: "seq_len"}, "logits": {0: "batch_size", 1: "seq_len"}},
        opset_version=14
    )
    onnx_size_mb = os.path.getsize(onnx_path) / (1024 * 1024)
    print(f"Modelo ONNX exportado exitosamente: {onnx_path} ({onnx_size_mb:.2f} MB)")
    
    # Benchmark ONNX Runtime on CPU
    print("\n--- Evaluando inferencia ONNX Runtime en CPU ---")
    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    
    def onnx_predict_tokens(tokens):
        enc = tokenizer(tokens, is_split_into_words=True, padding="max_length", max_length=MAX_LEN, truncation=True, return_tensors="np")
        word_ids = tokenizer(tokens, is_split_into_words=True, padding="max_length", max_length=MAX_LEN, truncation=True).word_ids()
        
        inputs = {
            "input_ids": enc["input_ids"].astype(np.int64),
            "attention_mask": enc["attention_mask"].astype(np.int64)
        }
        logits = session.run(["logits"], inputs)[0][0] # shape (MAX_LEN, num_labels)
        pred_ids = np.argmax(logits, axis=-1)
        
        # Align predictions to original tokens
        token_tags = []
        seen_words = set()
        for idx, w_id in enumerate(word_ids):
            if w_id is not None and w_id < len(tokens) and w_id not in seen_words:
                token_tags.append(id2label[pred_ids[idx]])
                seen_words.add(w_id)
                
        # Pad if needed
        while len(token_tags) < len(tokens):
            token_tags.append("O")
            
        first_names, second_names, pat_surnames, mat_surnames, company_tokens = [], [], [], [], []
        for t, tag in zip(tokens, token_tags):
            tag_type = tag.split("-")[-1] if "-" in tag else tag
            if tag_type == "COMPANY":
                company_tokens.append(t)
            elif tag_type == "FIRST_NAME":
                first_names.append(t)
            elif tag_type == "SECOND_NAME":
                second_names.append(t)
            elif tag_type == "PAT_SURNAME":
                pat_surnames.append(t)
            elif tag_type == "MAT_SURNAME":
                mat_surnames.append(t)
                
        is_comp = len(company_tokens) > (len(first_names) + len(pat_surnames) + len(mat_surnames))
        if is_comp:
            return True, " ".join(tokens), "", ""
        else:
            return False, " ".join(first_names + second_names), " ".join(pat_surnames), " ".join(mat_surnames)
            
    # Measure ONNX latency on CPU
    test_sample = ["MARIA", "DEL", "CARMEN", "CASTRO", "BELTRAN"]
    latencies = []
    for _ in range(100):
        t0 = time.time()
        onnx_predict_tokens(test_sample)
        latencies.append((time.time() - t0) * 1000) # ms
    avg_latency_ms = np.mean(latencies[10:])
    print(f"Latencia promedio ONNX en CPU: {avg_latency_ms:.2f} ms por nombre")
    
    # Real ground truth evaluation
    real_acc, total = evaluate_on_real_ground_truth(onnx_predict_tokens)
    print(f"Exactitud en Ground Truth Real (ONNX): {real_acc:.2f}% ({total} casos)")
    
    results = {
        "model": "Transformer (MiniLM-L12 -> ONNX)",
        "train_time_sec": train_duration,
        "latency_ms": avg_latency_ms,
        "onnx_size_mb": onnx_size_mb,
        "real_gt_accuracy": real_acc
    }
    with open(os.path.join(LAB_DIR, "transformer_benchmark_results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)
    print("\nBenchmark de Transformer (ONNX) concluido.")

if __name__ == "__main__":
    main()
