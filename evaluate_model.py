import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import config
from model import CXRAutoencoder, ECGAutoencoder, PCGAutoencoder, load_autoencoder


class LabeledDataset(Dataset):
    def __init__(self, data_path, label_path):
        self.data = np.load(data_path, mmap_mode="r")
        self.labels = np.load(label_path)
        if len(self.data) != len(self.labels):
            raise ValueError(
                f"Data/label length mismatch: {data_path} has {len(self.data)}, "
                f"but {label_path} has {len(self.labels)}."
            )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return {
            "data": torch.tensor(self.data[index], dtype=torch.float32),
            "label": int(self.labels[index]),
        }


def compute_errors(model, data_loader, device, modality):
    labels = []
    errors = []
    model.eval()

    with torch.no_grad():
        for batch in tqdm(data_loader, desc=f"Evaluating {modality.upper()}"):
            inputs = batch["data"].to(device)
            if modality == "cxr" and inputs.ndim == 3:
                inputs = inputs.unsqueeze(1)

            reconstruction = model(inputs)
            reduce_dims = (1, 2, 3) if modality == "cxr" else (1, 2)
            batch_errors = torch.mean(
                (reconstruction - inputs) ** 2, dim=reduce_dims
            )

            labels.extend(batch["label"].cpu().numpy())
            errors.extend(batch_errors.cpu().numpy())

    return np.asarray(labels, dtype=int), np.asarray(errors, dtype=float)


def fpr_at_sensitivity(labels, scores, target=0.90):
    fpr, tpr, _ = roc_curve(labels, scores)
    eligible = np.flatnonzero(tpr >= target)
    return float(np.min(fpr[eligible])) if len(eligible) else 1.0


def compute_metrics(labels, errors, threshold):
    predictions = (errors > threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    has_both_classes = len(np.unique(labels)) == 2

    metrics = {
        "samples": int(len(labels)),
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(labels, predictions)),
        "sensitivity": float(recall_score(labels, predictions, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if tn + fp else 0.0,
        "ppv": float(precision_score(labels, predictions, zero_division=0)),
        "npv": float(tn / (tn + fn)) if tn + fn else 0.0,
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "normal_error_mean": float(np.mean(errors[labels == 0])),
        "abnormal_error_mean": float(np.mean(errors[labels == 1])),
    }

    if has_both_classes:
        metrics.update(
            {
                "roc_auc_mse": float(roc_auc_score(labels, errors)),
                "roc_auc_inverse_mse": float(roc_auc_score(labels, -errors)),
                "pr_auc_mse": float(average_precision_score(labels, errors)),
                "fpr_at_90_sensitivity": fpr_at_sensitivity(labels, errors),
            }
        )
    else:
        metrics.update(
            {
                "roc_auc_mse": None,
                "roc_auc_inverse_mse": None,
                "pr_auc_mse": None,
                "fpr_at_90_sensitivity": None,
            }
        )
    return metrics, predictions


def plot_roc(labels, errors, modality, output_path):
    if len(np.unique(labels)) < 2:
        return
    fpr, tpr, _ = roc_curve(labels, errors)
    roc_auc = roc_auc_score(labels, errors)
    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, linewidth=2, label=f"MSE AUC = {roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"{modality.upper()} ROC Curve")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_error_distribution(labels, errors, threshold, modality, output_path):
    plt.figure(figsize=(7, 5))
    plt.hist(errors[labels == 0], bins=40, alpha=0.6, density=True, label="Normal")
    plt.hist(errors[labels == 1], bins=40, alpha=0.6, density=True, label="Abnormal")
    plt.axvline(threshold, color="red", linestyle="--", label=f"Threshold {threshold:.4f}")
    plt.xlabel("Reconstruction Error (MSE)")
    plt.ylabel("Density")
    plt.title(f"{modality.upper()} Error Distribution")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_confusion(labels, predictions, modality, output_path):
    matrix = confusion_matrix(labels, predictions, labels=[0, 1])
    plt.figure(figsize=(5, 4))
    plt.imshow(matrix, cmap="Blues")
    plt.title(f"{modality.upper()} Confusion Matrix")
    plt.xticks([0, 1], ["Normal", "Abnormal"])
    plt.yticks([0, 1], ["Normal", "Abnormal"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    for row in range(2):
        for column in range(2):
            plt.text(column, row, str(matrix[row, column]), ha="center", va="center")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def main():
    project_root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Evaluate each modality independently.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_root / "evaluation_results",
    )
    args = parser.parse_args()

    device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")
    threshold_data = np.load(project_root / "thresholds.npz")
    thresholds = {
        "ecg": float(threshold_data["ecg"]),
        "pcg": float(threshold_data["pcg"]),
        "cxr": float(threshold_data["cxr"]),
    }
    modalities = (
        ("ecg", ECGAutoencoder, "ecg_ae.pth", "ecg_data.npy", "ecg_labels.npy"),
        ("pcg", PCGAutoencoder, "pcg_ae.pth", "pcg_data.npy", "pcg_labels.npy"),
        ("cxr", CXRAutoencoder, "cxr_ae.pth", "cxr_data.npy", "cxr_labels.npy"),
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    print(f"Using device: {device}")

    for name, model_cls, weight_file, data_file, label_file in modalities:
        model = load_autoencoder(model_cls, project_root / weight_file, device)
        dataset = LabeledDataset(
            Path(config.OUTPUT_DIR) / data_file,
            Path(config.OUTPUT_DIR) / label_file,
        )
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
        labels, errors = compute_errors(model, loader, device, name)
        metrics, predictions = compute_metrics(labels, errors, thresholds[name])
        metrics["checkpoint_architecture"] = model.checkpoint_architecture
        results[name] = metrics

        plot_roc(labels, errors, name, args.output_dir / f"{name}_roc_curve.png")
        plot_error_distribution(
            labels,
            errors,
            thresholds[name],
            name,
            args.output_dir / f"{name}_error_distribution.png",
        )
        plot_confusion(
            labels,
            predictions,
            name,
            args.output_dir / f"{name}_confusion_matrix.png",
        )

        print(f"\n{name.upper()} ({model.checkpoint_architecture})")
        for key, value in metrics.items():
            print(f"  {key}: {value}")
        if (
            metrics["roc_auc_mse"] is not None
            and metrics["roc_auc_inverse_mse"] > metrics["roc_auc_mse"]
        ):
            print("  WARNING: -MSE ranks abnormalities better than MSE; inspect score direction.")

    with (args.output_dir / "evaluation_summary.json").open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)
    print(f"\nResults saved to {args.output_dir}")


if __name__ == "__main__":
    main()
