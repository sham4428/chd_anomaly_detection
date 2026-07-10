import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from tqdm import tqdm
from model import ECGAutoencoder, PCGAutoencoder
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

def get_model_errors(model, dataset, device, modality):
    loader = DataLoader(dataset, batch_size=32, shuffle=False)
    all_errors = []
    all_labels = []
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Computing {modality} errors"):
            x = batch["data"].to(device)
            labels = batch["label"].numpy()
            if modality == "cxr":
                x = x.unsqueeze(1)
            recon = model(x)
            if modality == "cxr":
                mse = torch.mean((recon - x) ** 2, dim=(1, 2, 3)).cpu().numpy()
            else:
                mse = torch.mean((recon - x) ** 2, dim=(1, 2)).cpu().numpy()
            all_errors.extend(mse)
            all_labels.extend(labels)
    return np.array(all_errors), np.array(all_labels)

def main():
    device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load thresholds
    thresholds = np.load("thresholds.npz")
    print("\nClinical Thresholds:")
    for key in thresholds:
        print(f"  {key}: {thresholds[key]:.6f}")
    
    # Load ECG
    print("\n--- ECG Model ---")
    ecg_model = ECGAutoencoder().to(device)
    ecg_model.load_state_dict(torch.load("ecg_ae.pth", map_location=device, weights_only=True))
    ecg_model.eval()
    ecg_dataset = LabeledDataset(
        os.path.join(config.OUTPUT_DIR, "ecg_data.npy"),
        os.path.join(config.OUTPUT_DIR, "ecg_labels.npy")
    )
    ecg_errors, ecg_labels = get_model_errors(ecg_model, ecg_dataset, device, "ecg")
    ecg_preds = (ecg_errors > thresholds["ecg"]).astype(int)
    
    # Load PCG
    print("\n--- PCG Model ---")
    pcg_model = PCGAutoencoder().to(device)
    pcg_model.load_state_dict(torch.load("pcg_ae.pth", map_location=device, weights_only=True))
    pcg_model.eval()
    pcg_dataset = LabeledDataset(
        os.path.join(config.OUTPUT_DIR, "pcg_data.npy"),
        os.path.join(config.OUTPUT_DIR, "pcg_labels.npy")
    )
    pcg_errors, pcg_labels = get_model_errors(pcg_model, pcg_dataset, device, "pcg")
    pcg_preds = (pcg_errors > thresholds["pcg"]).astype(int)
    
    # Check sample counts
    print(f"\nECG samples: {len(ecg_errors)}, PCG samples: {len(pcg_errors)}")
    
    # For parallel strategy, let's evaluate each separately first
    def compute_metrics(y_true, y_pred):
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp)
        return {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred),
            "recall": recall_score(y_true, y_pred),
            "f1": f1_score(y_true, y_pred),
            "specificity": specificity,
            "sensitivity": recall_score(y_true, y_pred),
            "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn)
        }
    
    print("\n--- ECG Performance ---")
    ecg_metrics = compute_metrics(ecg_labels, ecg_preds)
    for k, v in ecg_metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")
    
    print("\n--- PCG Performance ---")
    pcg_metrics = compute_metrics(pcg_labels, pcg_preds)
    for k, v in pcg_metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")
    
    print("\n" + "="*60)
    print("EVALUATION COMPLETE")
    print("="*60)
    
    print("\n💡 Note:")
    print("ECG and PCG datasets are different sizes, can't compute parallel combination directly.")
    print("Clinical strategy: use either ECG or PCG as primary screening tool.")

if __name__ == "__main__":
    main()
