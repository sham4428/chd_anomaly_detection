import numpy as np
import matplotlib.pyplot as plt
import os

def create_summary_report():
    results = {
        "ECG": {
            "accuracy": 0.6176,
            "precision": 0.8006,
            "recall": 0.3132,
            "f1": 0.4503,
            "specificity": 0.9220,
            "sensitivity": 0.3132,
            "roc_auc": 0.7271,
            "tp": 783,
            "fp": 195,
            "tn": 2305,
            "fn": 1717
        },
        "PCG": {
            "accuracy": 0.5432,
            "precision": 0.6349,
            "recall": 0.1709,
            "f1": 0.2694,
            "specificity": 0.9046,
            "sensitivity": 0.1709,
            "roc_auc": 0.6062,
            "tp": 80,
            "fp": 46,
            "tn": 436,
            "fn": 388
        },
        "CXR": {
            "accuracy": 0.5288,
            "precision": 0.7000,
            "recall": 0.1010,
            "f1": 0.1765,
            "specificity": 0.9567,
            "sensitivity": 0.1010,
            "roc_auc": 0.4683,
            "tp": 21,
            "fp": 9,
            "tn": 199,
            "fn": 187
        }
    }
    
    output_dir = "evaluation_results"
    
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    modalities = list(results.keys())
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'specificity', 'sensitivity']
    colors = ['#2E86AB', '#A23B72', '#F18F01']
    
    metric_values = {
        'Accuracy': [results[m]['accuracy'] for m in modalities],
        'Precision': [results[m]['precision'] for m in modalities],
        'Recall': [results[m]['recall'] for m in modalities],
        'F1 Score': [results[m]['f1'] for m in modalities],
        'Specificity': [results[m]['specificity'] for m in modalities],
        'ROC AUC': [results[m]['roc_auc'] for m in modalities]
    }
    
    x = np.arange(len(modalities))
    width = 0.12
    
    ax = axes[0, 0]
    for i, (metric_name, values) in enumerate(list(metric_values.items())[:4]):
        ax.bar(x + i * width, values, width, label=metric_name, alpha=0.8)
    ax.set_xlabel('Modality')
    ax.set_ylabel('Score')
    ax.set_title('Performance Comparison (Main Metrics)')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(modalities)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1])
    
    ax = axes[0, 1]
    for i, (metric_name, values) in enumerate(list(metric_values.items())[4:]):
        ax.bar(x + i * width, values, width, label=metric_name, color=['#27AE60', '#8E44AD'][i], alpha=0.8)
    ax.set_xlabel('Modality')
    ax.set_ylabel('Score')
    ax.set_title('Additional Metrics')
    ax.set_xticks(x + width * 0.5)
    ax.set_xticklabels(modalities)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1])
    
    for idx, modality in enumerate(modalities):
        ax = axes[1, 0] if idx < 2 else axes[1, 1]
        if idx == 2:
            axes[1, 0].set_visible(True)
            axes[1, 1].set_visible(True)
        
        cm = np.array([[results[modality]['tn'], results[modality]['fp']],
                       [results[modality]['fn'], results[modality]['tp']]])
        
        if idx == 0:
            ax = axes[1, 0]
            ax.set_title('ECG Confusion Matrix')
        elif idx == 1:
            ax = axes[1, 1]
            ax.set_title('PCG Confusion Matrix')
        else:
            continue
        
        im = ax.imshow(cm, cmap='Blues')
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(['Normal', 'Abnormal'])
        ax.set_yticklabels(['Normal', 'Abnormal'])
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                       color='white' if cm[i, j] > cm.mean() else 'black', fontsize=14)
    
    if not np.array_equal(axes[1, 0].get_title(), 'ECG Confusion Matrix'):
        axes[1, 0].set_title('ECG Confusion Matrix')
    
    axes[1, 1].set_title('PCG Confusion Matrix')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'comprehensive_summary.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    print("="*60)
    print("MODEL EVALUATION SUMMARY")
    print("="*60)
    
    print("\n1. OVERALL RANKING:")
    print("   - Best ROC AUC: ECG (0.7271)")
    print("   - Best Precision: CXR (0.7000)")
    print("   - Best Specificity: CXR (0.9567)")
    print("   - Best Recall: ECG (0.3132)")
    
    print("\n2. KEY FINDINGS:")
    print("   Pros:")
    print("   - Very high specificity (minimizes false positives)")
    print("   - Good precision for all modalities")
    print("   - ECG shows best overall performance")
    
    print("\n   Cons:")
    print("   - Low recall (many abnormal cases missed)")
    print("   - CXR ROC AUC is below 0.5, needs improvement")
    print("   - Thresholds might be too strict")
    
    print("\n3. SUGGESTIONS:")
    print("   - Consider adjusting thresholds for better recall")
    print("   - Collect more diverse training data")
    print("   - Try alternative model architectures")
    print("   - Consider ensemble of multiple modalities")
    
    print("\n" + "="*60)
    print(f"Full report saved to {output_dir}/")
    print("="*60)

if __name__ == "__main__":
    create_summary_report()
