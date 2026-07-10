import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
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

def plot_roc(y_true, errors, name, save_dir):
    fpr, tpr, _ = roc_curve(y_true, errors)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(10, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC AUC = {roc_auc:.3f}')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {name.upper()}')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, f"{name}_roc_full.png"), dpi=150, bbox_inches='tight')
    plt.close()
    return roc_auc

def plot_error_distribution(errors_normal, errors_abnormal, threshold, name, save_dir):
    plt.figure(figsize=(12, 6))
    plt.hist(errors_normal, bins=50, alpha=0.5, label='Normal', density=True)
    plt.hist(errors_abnormal, bins=50, alpha=0.5, label='Abnormal', density=True)
    plt.axvline(x=threshold, color='red', linestyle='--', label=f'Threshold: {threshold:.6f}')
    plt.xlabel('Reconstruction Error (MSE)')
    plt.ylabel('Density')
    plt.title(f'Error Distribution - {name.upper()}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(save_dir, f"{name}_error_dist_full.png"), dpi=150, bbox_inches='tight')
    plt.close()

def evaluate_modality(name, model_cls, model_path, data_file, label_file, threshold, device, save_dir):
    print(f"\n{'='*60}")
    print(f"Evaluating {name.upper()}")
    print(f"{'='*60}")
    
    model = model_cls().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    
    dataset = LabeledDataset(
        os.path.join(config.OUTPUT_DIR, data_file),
        os.path.join(config.OUTPUT_DIR, label_file)
    )
    loader = DataLoader(dataset, batch_size=32, shuffle=False)
    
    all_labels = []
    all_preds = []
    all_errors = []
    
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Testing {name}"):
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
    
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "specificity": tn / (tn + fp),
        "sensitivity": recall_score(y_true, y_pred),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
        "roc_auc": plot_roc(y_true, errors, name, save_dir)
    }
    
    plot_error_distribution(
        errors[y_true == 0], errors[y_true == 1], threshold, name, save_dir
    )
    
    print("\nMetrics:")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")
    
    return metrics, y_true, y_pred, errors

def main():
    device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    save_dir = "test_results"
    os.makedirs(save_dir, exist_ok=True)
    
    # Load optimized thresholds
    thresholds = np.load("thresholds.npz")
    print("\nOptimized Clinical Thresholds:")
    for key in thresholds:
        print(f"  {key}: {thresholds[key]:.6f}")
    
    modalities = [
        ("ecg", ECGAutoencoder, "ecg_ae.pth", "ecg_data.npy", "ecg_labels.npy"),
        ("pcg", PCGAutoencoder, "pcg_ae.pth", "pcg_data.npy", "pcg_labels.npy"),
        ("cxr", CXRAutoencoder, "cxr_ae.pth", "cxr_data.npy", "cxr_labels.npy"),
    ]
    
    all_metrics = {}
    
    for name, model_cls, model_path, data_file, label_file in modalities:
        metrics, _, _, _ = evaluate_modality(
            name, model_cls, model_path, data_file, label_file, thresholds[name], device, save_dir
        )
        all_metrics[name] = metrics
    
    # Save summary report
    report_path = os.path.join(save_dir, "full_test_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=== FULL TEST REPORT (Clinical Optimized) ===\n")
        f.write("="*60 + "\n\n")
        f.write("Optimized Thresholds:\n")
        for key in thresholds:
            f.write(f"  {key}: {thresholds[key]:.6f}\n")
        
        for name in ["ecg", "pcg", "cxr"]:
            f.write(f"\n{name.upper()} Model:\n")
            f.write("-"*40 + "\n")
            for k, v in all_metrics[name].items():
                if isinstance(v, float):
                    f.write(f"  {k}: {v:.4f}\n")
                else:
                    f.write(f"  {k}: {v}\n")
        
    print(f"\n{'='*60}")
    print("✅ TEST COMPLETE!")
    print(f"{'='*60}")
    print(f"Results saved to {save_dir}/")
    print(f"  - full_test_report.txt")
    for name in ["ecg", "pcg", "cxr"]:
        print(f"  - {name}_error_dist_full.png")
        print(f"  - {name}_roc_full.png")

if __name__ == "__main__":
    main()
