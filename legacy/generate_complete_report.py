import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, confusion_matrix, roc_curve, auc)
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

def plot_roc_curve(y_true, errors, threshold, modality, save_path):
    fpr, tpr, _ = roc_curve(y_true, errors)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(10, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (area = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Receiver Operating Characteristic - {modality}')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    return roc_auc

def plot_error_distribution(errors_normal, errors_abnormal, threshold, modality, save_path):
    plt.figure(figsize=(12, 6))
    plt.hist(errors_normal, bins=50, alpha=0.5, label='Normal', density=True)
    plt.hist(errors_abnormal, bins=50, alpha=0.5, label='Abnormal', density=True)
    plt.axvline(x=threshold, color='red', linestyle='--', 
                label=f'Threshold: {threshold:.6f}')
    plt.xlabel('Reconstruction Error (MSE)')
    plt.ylabel('Density')
    plt.title(f'Error Distribution - {modality}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def plot_comprehensive_summary(all_metrics, save_path):
    modalities = list(all_metrics.keys())
    metrics = ['recall', 'precision', 'specificity', 'accuracy']
    metric_names = ['Recall', 'Precision', 'Specificity', 'Accuracy']
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.ravel()
    
    colors = ['#4299e1', '#48bb78', '#ed8936', '#9f7aea']
    
    for i, (metric, name) in enumerate(zip(metrics, metric_names)):
        values = [all_metrics[mod][metric] * 100 for mod in modalities]
        bars = axes[i].bar(modalities, values, color=colors, alpha=0.8)
        axes[i].set_title(f'{name} Comparison', fontsize=14, fontweight='bold')
        axes[i].set_ylabel('Percentage (%)', fontsize=12)
        axes[i].set_ylim([0, 100])
        axes[i].grid(True, alpha=0.3, axis='y')
        
        for bar in bars:
            height = bar.get_height()
            axes[i].text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.1f}%',
                        ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
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
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn)
    }
    
    metrics["roc_auc"] = plot_roc_curve(
        y_true, errors, threshold, name.upper(),
        os.path.join(save_dir, f"{name}_roc_curve.png")
    )
    
    plot_error_distribution(
        errors[y_true == 0], errors[y_true == 1], threshold, name.upper(),
        os.path.join(save_dir, f"{name}_error_dist.png")
    )
    
    print("\nMetrics:")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")
    
    return metrics

def generate_html_report(all_metrics, thresholds, save_dir):
    html = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>CHD Anomaly Detection - Complete Evaluation Report</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        .container { max-width: 1200px; margin: 0 auto; }
        .header {
            background: white; padding: 30px; border-radius: 12px;
            box-shadow: 0 8px 32px rgba(0,0,0,0.1); margin-bottom: 30px; text-align: center;
        }
        .header h1 { color: #2d3748; font-size: 2.5em; margin-bottom: 10px; }
        .header p { color: #718096; font-size: 1.1em; }
        .summary-cards {
            display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 20px; margin-bottom: 30px;
        }
        .card {
            background: white; padding: 25px; border-radius: 12px;
            box-shadow: 0 4px 16px rgba(0,0,0,0.1); transition: transform 0.3s;
        }
        .card:hover { transform: translateY(-5px); box-shadow: 0 8px 24px rgba(0,0,0,0.15); }
        .card h3 {
            color: #4a5568; margin-bottom: 15px; font-size: 1.3em;
            border-bottom: 2px solid #e2e8f0; padding-bottom: 10px;
        }
        .metric {
            display: flex; justify-content: space-between; margin: 10px 0;
            padding: 8px 0; border-bottom: 1px solid #f7fafc;
        }
        .metric-label { color: #718096; font-weight: 500; }
        .metric-value { color: #2d3748; font-weight: bold; }
        .good { color: #48bb78; }
        .moderate { color: #ed8936; }
        .poor { color: #f56565; }
        .section {
            background: white; padding: 30px; border-radius: 12px;
            box-shadow: 0 4px 16px rgba(0,0,0,0.1); margin-bottom: 30px;
        }
        .section h2 { color: #2d3748; margin-bottom: 20px; font-size: 1.8em; }
        .images-grid {
            display: grid; grid-template-columns: repeat(auto-fit, minmax(500px, 1fr));
            gap: 20px;
        }
        .image-card {
            background: #f7fafc; padding: 15px; border-radius: 8px;
        }
        .image-card img { width: 100%; border-radius: 6px; }
        .image-card h4 { color: #4a5568; margin-bottom: 10px; text-align: center; }
        .thresholds {
            background: #f0fff4; padding: 20px; border-radius: 8px; margin-bottom: 20px;
            border-left: 4px solid #48bb78;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>💓 CHD Anomaly Detection - Complete Evaluation Report</h1>
            <p>Clinical Optimized Version</p>
        </div>
        
        <div class="section">
            <h2>🎯 Optimized Thresholds</h2>
            <div class="thresholds">
"""
    
    for name in thresholds:
        html += f"                <p><strong>{name.upper()}:</strong> {thresholds[name]:.6f}</p>\n"
    
    html += """            </div>
        </div>
        
        <div class="section">
            <h2>📊 Summary Performance</h2>
            <div class="images-grid">
                <div class="image-card">
                    <h4>Comprehensive Summary</h4>
                    <img src="comprehensive_summary.png" alt="Comprehensive Summary">
                </div>
            </div>
        </div>
"""
    
    modalities = ["ecg", "pcg", "cxr"]
    modality_names = ["ECG", "PCG", "CXR"]
    
    for name, disp_name in zip(modalities, modality_names):
        metrics = all_metrics[name]
        
        recall_class = "good" if metrics["recall"] >= 0.9 else ("moderate" if metrics["recall"] >= 0.7 else "poor")
        
        html += f"""
        <div class="section">
            <h2>🔬 {disp_name} Model Performance</h2>
            <div class="summary-cards">
                <div class="card">
                    <h3>Key Metrics</h3>
                    <div class="metric">
                        <span class="metric-label">Accuracy</span>
                        <span class="metric-value">{metrics['accuracy']*100:.2f}%</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">Precision</span>
                        <span class="metric-value">{metrics['precision']*100:.2f}%</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">Recall (Sensitivity)</span>
                        <span class="metric-value {recall_class}">{metrics['recall']*100:.2f}%</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">F1 Score</span>
                        <span class="metric-value">{metrics['f1']*100:.2f}%</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">Specificity</span>
                        <span class="metric-value">{metrics['specificity']*100:.2f}%</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">ROC AUC</span>
                        <span class="metric-value">{metrics['roc_auc']:.4f}</span>
                    </div>
                </div>
                <div class="card">
                    <h3>Confusion Matrix</h3>
                    <div class="metric">
                        <span class="metric-label">True Positive (TP)</span>
                        <span class="metric-value">{metrics['tp']}</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">False Positive (FP)</span>
                        <span class="metric-value">{metrics['fp']}</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">True Negative (TN)</span>
                        <span class="metric-value">{metrics['tn']}</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">False Negative (FN)</span>
                        <span class="metric-value">{metrics['fn']}</span>
                    </div>
                </div>
            </div>
            <div class="images-grid">
                <div class="image-card">
                    <h4>Error Distribution</h4>
                    <img src="{name}_error_dist.png" alt="{disp_name} Error Distribution">
                </div>
                <div class="image-card">
                    <h4>ROC Curve</h4>
                    <img src="{name}_roc_curve.png" alt="{disp_name} ROC Curve">
                </div>
            </div>
        </div>
"""
    
    html += """
    </div>
</body>
</html>
"""
    
    with open(os.path.join(save_dir, "full_report.html"), "w", encoding="utf-8") as f:
        f.write(html)

def generate_text_report(all_metrics, thresholds, save_dir):
    with open(os.path.join(save_dir, "evaluation_summary.txt"), "w", encoding="utf-8") as f:
        f.write("CHD Anomaly Detection - Complete Evaluation Report (Clinical Optimized)\n")
        f.write("="*70 + "\n\n")
        f.write("Optimized Thresholds:\n")
        for name in thresholds:
            f.write(f"  {name}: {thresholds[name]:.6f}\n")
        
        for name in all_metrics:
            f.write(f"\n\n{name.upper()} Model:\n")
            f.write("-"*40 + "\n")
            for k, v in all_metrics[name].items():
                if isinstance(v, float):
                    f.write(f"  {k}: {v:.4f}\n")
                else:
                    f.write(f"  {k}: {v}\n")

def main():
    device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    save_dir = "test_results_complete"
    os.makedirs(save_dir, exist_ok=True)
    
    thresholds = np.load("thresholds.npz")
    print("\nOptimized Thresholds:")
    for key in thresholds:
        print(f"  {key}: {thresholds[key]:.6f}")
    
    modalities = [
        ("ecg", ECGAutoencoder, "ecg_ae.pth", "ecg_data.npy", "ecg_labels.npy"),
        ("pcg", PCGAutoencoder, "pcg_ae.pth", "pcg_data.npy", "pcg_labels.npy"),
        ("cxr", CXRAutoencoder, "cxr_ae.pth", "cxr_data.npy", "cxr_labels.npy"),
    ]
    
    all_metrics = {}
    
    for name, model_cls, model_path, data_file, label_file in modalities:
        all_metrics[name] = evaluate_modality(
            name, model_cls, model_path, data_file, label_file, 
            thresholds[name], device, save_dir
        )
    
    plot_comprehensive_summary(all_metrics, os.path.join(save_dir, "comprehensive_summary.png"))
    generate_html_report(all_metrics, thresholds, save_dir)
    generate_text_report(all_metrics, thresholds, save_dir)
    
    print(f"\n{'='*70}")
    print("✅ COMPLETE EVALUATION FINISHED!")
    print(f"{'='*70}")
    print(f"All results saved to: {save_dir}/")
    print(f"  - full_report.html (Web Report)")
    print(f"  - evaluation_summary.txt")
    print(f"  - comprehensive_summary.png")
    for name in all_metrics:
        print(f"  - {name}_error_dist.png")
        print(f"  - {name}_roc_curve.png")

if __name__ == "__main__":
    main()
