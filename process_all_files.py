import os
import re
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib

def main():
    parquet_path = "files_dataset.parquet"
    if not os.path.exists(parquet_path):
        raise FileNotFoundError("files_dataset.parquet not found. Run build_dataset.py first!")
        
    df_matched = pd.read_parquet(parquet_path)
    
    print("Training the winner model (TF-IDF + Logistic Regression) on matched data...")
    X_train_text = df_matched["extracted_text"].values
    y_train = df_matched["label_movement"].values
    
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_train_vec = vectorizer.fit_transform(X_train_text)
    
    clf = LogisticRegression(class_weight="balanced", random_state=42)
    clf.fit(X_train_vec, y_train)
    
    # Save the trained model and vectorizer
    os.makedirs("models", exist_ok=True)
    joblib.dump(vectorizer, "models/tfidf_vectorizer.joblib")
    joblib.dump(clf, "models/logistic_regression_model.joblib")
    
    # Now scan the directory for ALL 116 PDFs and fetch their text from transcriptions/ cache
    print("\nScanning for all PDFs in workspace...")
    all_records = []
    
    # Map matched records by PDF filename for easy lookup
    matched_lookup = {row["pdf_name"]: row.to_dict() for idx, row in df_matched.iterrows()}
    
    for root, dirs, files in os.walk("files"):
        for file in files:
            if file.lower().endswith(".pdf"):
                path = os.path.join(root, file)
                basename = os.path.splitext(file)[0]
                cache_path = os.path.join("transcriptions", f"{basename}.txt")
                
                # Load cached transcription text
                text = ""
                if os.path.exists(cache_path):
                    with open(cache_path, "r", encoding="utf-8") as f:
                        text = f.read()
                
                # Check if we have ground truth from alignment
                matched = matched_lookup.get(file)
                
                all_records.append({
                    "pdf_name": file,
                    "pdf_path": path,
                    "pdf_dir": os.path.basename(root),
                    "extracted_text": text,
                    "label_movement": matched["label_movement"] if matched else "N/A",
                    "label_nombre": matched["label_nombre"] if matched else "N/A",
                    "label_motivo": matched["label_motivo"] if matched else "N/A",
                    "match_type": matched["match_type"] if matched else "unmatched"
                })
                
    df_all = pd.DataFrame(all_records)
    print(f"Found total of {len(df_all)} PDFs in files/.")
    
    # Predict on all
    X_all_vec = vectorizer.transform(df_all["extracted_text"])
    preds = clf.predict(X_all_vec)
    probs = clf.predict_proba(X_all_vec)
    
    df_all["predicted_movement"] = preds
    pred_probs = [probs[i, 1] if p == "BAJA" else probs[i, 0] for i, p in enumerate(preds)]
    df_all["confidence_score"] = pred_probs
    
    # Calculate correctness where ground truth exists
    def check_correctness(row):
        if row["label_movement"] == "N/A":
            return "N/A"
        return str(row["label_movement"] == row["predicted_movement"])
        
    df_all["prediction_correct"] = df_all.apply(check_correctness, axis=1)
    
    # Reorder columns for output
    cols = [
        "pdf_name", "pdf_dir", "label_movement", "predicted_movement", 
        "confidence_score", "prediction_correct", "label_nombre", 
        "label_motivo", "match_type"
    ]
    df_output = df_all[cols].copy()
    
    # Save results
    df_output.to_csv("final_predictions.csv", index=False)
    df_output.to_parquet("final_predictions.parquet", index=False)
    
    print("\n--- Final Predictions Summary ---")
    print(f"Total PDFs: {len(df_output)}")
    print(f"Predicted ALTA: {sum(df_output['predicted_movement'] == 'ALTA')}")
    print(f"Predicted BAJA: {sum(df_output['predicted_movement'] == 'BAJA')}")
    
    unmatched_rows = df_output[df_output["match_type"] == "unmatched"]
    if len(unmatched_rows) > 0:
        print("\nPredictions for unmatched files:")
        for idx, r in unmatched_rows.iterrows():
            print(f"  - {r['pdf_name']}: Predicted {r['predicted_movement']} with {r['confidence_score']*100:.2f}% confidence.")
            
    mismatches = df_output[df_output["prediction_correct"] == "False"]
    if len(mismatches) == 0:
        print("\nAll labeled predictions match the ground-truth metadata exactly!")
    else:
        print(f"\nWarning: Found {len(mismatches)} mismatches between predictions and CSV labels:")
        print(mismatches[["pdf_name", "label_movement", "predicted_movement", "confidence_score"]])

if __name__ == "__main__":
    main()
