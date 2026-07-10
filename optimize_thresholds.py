import argparse
import os

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

import config
from model import CXRAutoencoder, ECGAutoencoder, PCGAutoencoder, load_autoencoder


class LabeledDataset(Dataset):
    def __init__(self, data_path, label_path):
        self.data = np.load(data_path, mmap_mode="r")
        self.labels = np.load(label_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {
            "data": torch.tensor(self.data[idx], dtype=torch.float32),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
        }


def compute_errors(name, model_cls, model_path, data_file, label_file, device):
    model = load_autoencoder(model_cls, model_path, device)
    print(f"Loaded architecture: {model.checkpoint_architecture}")

    dataset = LabeledDataset(
        os.path.join(config.OUTPUT_DIR, data_file),
        os.path.join(config.OUTPUT_DIR, label_file),
    )
    data_loader = DataLoader(dataset, batch_size=32, shuffle=False)

    all_labels = []
    all_errors = []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc=f"Computing errors for {name}"):
            x = batch["data"].to(device)
            labels = batch["label"].numpy()

            if name == "cxr":
                x = x.unsqueeze(1)

            recon = model(x)

            if name == "cxr":
                mse = torch.mean((recon - x) ** 2, dim=(1, 2, 3)).cpu().numpy()
            else:
                mse = torch.mean((recon - x) ** 2, dim=(1, 2)).cpu().numpy()

            all_labels.extend(labels)
            all_errors.extend(mse)

    return np.array(all_labels), np.array(all_errors)


def stratified_split(labels, test_size=0.5, random_state=42):
    rng = np.random.default_rng(random_state)
    calibration_indices = []
    test_indices = []

    for label in np.unique(labels):
        label_indices = np.flatnonzero(labels == label)
        rng.shuffle(label_indices)
        test_count = max(1, int(np.ceil(len(label_indices) * test_size)))
        test_indices.extend(label_indices[:test_count])
        calibration_indices.extend(label_indices[test_count:])

    return np.array(calibration_indices), np.array(test_indices)


def confusion_counts(labels, preds):
    labels = labels.astype(int)
    preds = preds.astype(int)
    tp = int(np.sum((labels == 1) & (preds == 1)))
    fp = int(np.sum((labels == 0) & (preds == 1)))
    tn = int(np.sum((labels == 0) & (preds == 0)))
    fn = int(np.sum((labels == 1) & (preds == 0)))
    return tp, fp, tn, fn


def safe_div(num, den):
    return float(num / den) if den else 0.0


def binary_metrics(labels, scores, threshold):
    preds = (scores > threshold).astype(int)
    tp, fp, tn, fn = confusion_counts(labels, preds)
    return {
        "threshold": float(threshold),
        "sensitivity": safe_div(tp, tp + fn),
        "recall": safe_div(tp, tp + fn),
        "specificity": safe_div(tn, tn + fp),
        "precision": safe_div(tp, tp + fp),
        "ppv": safe_div(tp, tp + fp),
        "npv": safe_div(tn, tn + fn),
        "accuracy": safe_div(tp + tn, len(labels)),
        "f1": safe_div(2 * tp, 2 * tp + fp + fn),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }


def roc_auc(labels, scores):
    labels = labels.astype(int)
    pos = int(np.sum(labels == 1))
    neg = int(np.sum(labels == 0))
    if pos == 0 or neg == 0:
        return float("nan")

    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)

    sorted_scores = scores[order]
    start = 0
    while start < len(scores):
        end = start + 1
        while end < len(scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        if end - start > 1:
            ranks[order[start:end]] = np.mean(np.arange(start + 1, end + 1))
        start = end

    rank_sum_pos = float(np.sum(ranks[labels == 1]))
    return (rank_sum_pos - pos * (pos + 1) / 2) / (pos * neg)


def pr_auc(labels, scores):
    labels = labels.astype(int)
    positives = int(np.sum(labels == 1))
    if positives == 0:
        return float("nan")

    order = np.argsort(-scores)
    sorted_labels = labels[order]
    tp = np.cumsum(sorted_labels == 1)
    fp = np.cumsum(sorted_labels == 0)
    recall = tp / positives
    precision = tp / np.maximum(tp + fp, 1)

    recall = np.concatenate([[0.0], recall])
    precision = np.concatenate([[1.0], precision])
    return float(np.trapz(precision, recall))


def fpr_at_sensitivity(labels, scores, target_sensitivity=0.90):
    thresholds = np.unique(scores)
    best_fpr = None
    for threshold in thresholds:
        metrics = binary_metrics(labels, scores, threshold)
        if metrics["sensitivity"] >= target_sensitivity:
            fpr = 1.0 - metrics["specificity"]
            best_fpr = fpr if best_fpr is None else min(best_fpr, fpr)
    return float(best_fpr) if best_fpr is not None else float("nan")


def find_threshold_for_recall(labels, errors, target_recall=0.90):
    best_threshold = None
    best_metrics = None

    for threshold in np.unique(errors):
        metrics = binary_metrics(labels, errors, threshold)
        if metrics["recall"] < target_recall:
            continue
        if best_metrics is None or metrics["specificity"] > best_metrics["specificity"]:
            best_threshold = threshold
            best_metrics = metrics

    if best_threshold is None:
        print(f"Warning: could not reach target recall {target_recall:.2f}; using max-recall threshold.")
        best_threshold = np.min(errors) - 1e-6
        best_metrics = binary_metrics(labels, errors, best_threshold)

    return float(best_threshold), best_metrics


def add_ranking_metrics(metrics, labels, errors):
    metrics["roc_auc_mse"] = float(roc_auc(labels, errors))
    metrics["roc_auc_inverse_mse"] = float(roc_auc(labels, -errors))
    metrics["pr_auc_mse"] = float(pr_auc(labels, errors))
    metrics["fpr_at_90_sensitivity"] = fpr_at_sensitivity(labels, errors, 0.90)
    return metrics


def format_metrics(prefix, metrics):
    lines = [prefix]
    for key in [
        "threshold",
        "sensitivity",
        "specificity",
        "precision",
        "npv",
        "accuracy",
        "f1",
        "roc_auc_mse",
        "roc_auc_inverse_mse",
        "pr_auc_mse",
        "fpr_at_90_sensitivity",
    ]:
        if key in metrics:
            lines.append(f"  {key}: {metrics[key]:.6f}")
    lines.append(
        f"  confusion_matrix: tp={metrics['tp']}, fp={metrics['fp']}, "
        f"tn={metrics['tn']}, fn={metrics['fn']}"
    )
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibration_size", type=float, default=0.5)
    parser.add_argument("--random_state", type=int, default=42)
    args = parser.parse_args()

    device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    modalities = [
        ("ecg", ECGAutoencoder, "ecg_ae.pth", "ecg_data.npy", "ecg_labels.npy", 0.90),
        ("pcg", PCGAutoencoder, "pcg_ae.pth", "pcg_data.npy", "pcg_labels.npy", 0.90),
        ("cxr", CXRAutoencoder, "cxr_ae.pth", "cxr_data.npy", "cxr_labels.npy", 0.80),
    ]

    optimized_thresholds = {}
    report_lines = [
        "Threshold optimization report",
        "Thresholds are selected only on calibration data; held-out test is report-only.",
        "",
    ]

    for name, model_cls, model_path, data_file, label_file, target_recall in modalities:
        print(f"\n{'=' * 60}")
        print(f"Optimizing threshold for {name.upper()}")
        print(f"{'=' * 60}")

        labels, errors = compute_errors(name, model_cls, model_path, data_file, label_file, device)
        calibration_indices, test_indices = stratified_split(
            labels,
            test_size=1.0 - args.calibration_size,
            random_state=args.random_state,
        )

        cal_labels = labels[calibration_indices]
        cal_errors = errors[calibration_indices]
        test_labels = labels[test_indices]
        test_errors = errors[test_indices]

        threshold, cal_metrics = find_threshold_for_recall(cal_labels, cal_errors, target_recall)
        cal_metrics = add_ranking_metrics(cal_metrics, cal_labels, cal_errors)
        test_metrics = add_ranking_metrics(binary_metrics(test_labels, test_errors, threshold), test_labels, test_errors)

        optimized_thresholds[name] = threshold

        print(f"Target recall on calibration: {target_recall:.2f}")
        print(f"Calibration samples: {len(cal_labels)}; held-out test samples: {len(test_labels)}")
        print(format_metrics("Calibration metrics:", cal_metrics))
        print(format_metrics("Held-out test metrics:", test_metrics))

        if test_metrics["roc_auc_inverse_mse"] > test_metrics["roc_auc_mse"]:
            print("Warning: -MSE AUC is higher on held-out test; inspect score direction before deployment.")

        report_lines.extend(
            [
                f"{name.upper()}",
                f"target_recall: {target_recall:.2f}",
                f"calibration_samples: {len(cal_labels)}",
                f"test_samples: {len(test_labels)}",
                format_metrics("calibration:", cal_metrics),
                format_metrics("held_out_test:", test_metrics),
                "",
            ]
        )

    np.savez("thresholds.npz", **optimized_thresholds)

    output_dir = "evaluation_results"
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "threshold_optimization_report.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    print(f"\n{'=' * 60}")
    print("OPTIMIZED THRESHOLDS")
    print(f"{'=' * 60}")
    for name, threshold in optimized_thresholds.items():
        print(f"  {name}: {threshold:.6f}")
    print("\nThresholds saved to thresholds.npz")
    print("Report saved to evaluation_results/threshold_optimization_report.txt")


if __name__ == "__main__":
    main()
