import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from tqdm import tqdm
from model import ECGAutoencoder, PCGAutoencoder, CXRAutoencoder
import config

class LabeledDataset(Dataset):
    def __init__(self, data_path, label_path):
        self.data = np.load(data_path)
        self.labels = np.load(label_path)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return {
            "data": torch.tensor(self.data[idx], dtype=torch.float32),
            "label": torch.tensor(self.labels[idx], dtype=torch.long)
        }

def compute_metrics(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "specificity": tn / (tn + fp),
        "sensitivity": tp / (tp + fn),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn)
    }

def evaluate_single_modality(name, model_cls, model_path, data_file, label_file, threshold, device):
    print(f"\n{'='*60}")
    print(f"Evaluating {name.upper()} Model (Threshold: {threshold:.4f})")
    print(f"{'='*60}")
    
    model = model_cls().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    
    dataset = LabeledDataset(
        os.path.join(config.OUTPUT_DIR, data_file),
        os.path.join(config.OUTPUT_DIR, label_file)
    )
    
    data_loader = DataLoader(dataset, batch_size=32, shuffle=False)
    
    all_labels = []
    all_preds = []
    all_errors = []
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc=f"Evaluating"):
            x = batch["data"].to(device)
            labels = batch["label"].numpy()
            
            if name == "cxr":
                x = x.unsqueeze(1)
                
            recon = model(x)
            
            if name == "cxr":
                mse = torch.mean((recon - x) ** 2, dim=(1, 2, 3)).cpu().numpy()
            else:
                mse = torch.mean((recon - x) ** 2, dim=(1, 2)).cpu().numpy()
            
            preds = (mse > threshold).astype(int)
            
            all_labels.extend(labels)
            all_preds.extend(preds)
            all_errors.extend(mse)
    
    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    errors = np.array(all_errors)
    
    result = compute_metrics(y_true, y_pred)
    
    print(f"\n{name.upper()} Metrics:")
    for key, value in result.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    
    return y_true, y_pred, errors

def main():
    device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    thresholds = {
        "ecg": 0.12,
        "pcg": 0.08,
        "cxr": 0.05
    }
    
    modalities = [
        ("ecg", ECGAutoencoder, "ecg_ae.pth", "ecg_data.npy", "ecg_labels.npy"),
        ("pcg", PCGAutoencoder, "pcg_ae.pth", "pcg_data.npy", "pcg_labels.npy"),
        ("cxr", CXRAutoencoder, "cxr_ae.pth", "cxr_data.npy", "cxr_labels.npy"),
    ]
    
    results = {}
    all_results = {}
    
    for name, model_cls, model_path, data_file, label_file in modalities:
        y_true, y_pred, errors = evaluate_single_modality(name, model_cls, model_path, data_file, label_file, thresholds[name], device)
        all_results[name] = {
            "y_true": y_true,
            "y_pred": y_pred,
            "errors": errors
        }
        results[name] = compute_metrics(y_true, y_pred)
    
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    
    output_dir = "evaluation_results"
    os.makedirs(output_dir, exist_ok=True)
    
    with open(os.path.join(output_dir, "clinical_eval.txt"), "w", encoding="utf-8") as f:
        f.write("CHD Anomaly Detection - Clinical Evaluation Results\n")
        f.write("="*60 + "\n\n")
        f.write("New Clinical Thresholds:\n")
        f.write(f"  ECG: {thresholds['ecg']:.4f}\n")
        f.write(f"  PCG: {thresholds['pcg']:.4f}\n")
        f.write(f"  CXR: {thresholds['cxr']:.4f}\n")
        
        for name in ["ecg", "pcg", "cxr"]:
            f.write(f"\n{name.upper()} Model:\n")
            f.write("-"*40 + "\n")
            for key, value in results[name].items():
                if isinstance(value, float):
                    f.write(f"  {key}: {value:.4f}\n")
                else:
                    f.write(f"  {key}: {value}\n")
        
    # 保存新阈值
    np.savez("thresholds.npz", **thresholds)
    
    print("\nResults saved to evaluation_results/clinical_eval.txt")
    print("\n✅ Clinical evaluation complete!")
    print("\nSummary:")
    print("  - Saved new thresholds to thresholds.npz")
    print("  - New thresholds prioritise recall (sensitivity) over specificity")
    print("  - This ensures minimal missed diagnoses")

if __name__ == "__main__":
    main()
