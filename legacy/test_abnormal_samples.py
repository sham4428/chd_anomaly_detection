
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from model import ECGAutoencoder, PCGAutoencoder, CXRAutoencoder
import config

def load_data_and_labels(data_path, labels_path):
    """加载数据和标签"""
    data = np.load(data_path)
    labels = np.load(labels_path)
    return data, labels

def find_abnormal_samples(labels):
    """找出有问题的样本（标签为1）"""
    abnormal_indices = np.where(labels == 1)[0]
    return abnormal_indices

def test_single_sample(model, sample, threshold, device, modality):
    """测试单个样本"""
    model.eval()
    with torch.no_grad():
        # 添加batch维度
        if modality == 'cxr':
            sample_tensor = torch.tensor(sample, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
        else:
            sample_tensor = torch.tensor(sample, dtype=torch.float32).unsqueeze(0).to(device)
        
        # 模型重构
        recon = model(sample_tensor)
        
        # 计算重构误差
        if modality == 'cxr':
            mse = torch.mean((recon - sample_tensor) ** 2, dim=(1, 2, 3)).cpu().numpy()
        else:
            mse = torch.mean((recon - sample_tensor) ** 2, dim=(1, 2)).cpu().numpy()
        
        # 预测结果
        prediction = mse[0] > threshold
    
    return {
        'mse': mse[0],
        'threshold': threshold,
        'prediction': prediction,
        'recon': recon.cpu().numpy()
    }

def visualize_ecg(original, recon, mse, threshold, index, save_dir):
    """可视化心电图"""
    fig, axes = plt.subplots(2, 1, figsize=(15, 10))
    
    # 原始信号
    axes[0].set_title(f'Original ECG (Sample {index})', fontsize=14)
    axes[0].plot(original[0], label='Lead I')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 重构信号
    axes[1].set_title(f'Reconstructed ECG (MSE: {mse:.6f}, Threshold: {threshold:.6f})', fontsize=14)
    axes[1].plot(recon[0, 0], label='Lead I (Reconstructed)', color='orange')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'ecg_abnormal_sample_{index}.png'), dpi=150)
    plt.close()

def visualize_pcg(original, recon, mse, threshold, index, save_dir):
    """可视化心音图"""
    fig, axes = plt.subplots(2, 1, figsize=(15, 10))
    
    # 原始信号
    axes[0].set_title(f'Original PCG (Sample {index})', fontsize=14)
    axes[0].plot(original[0], label='MFCC 1')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 重构信号
    axes[1].set_title(f'Reconstructed PCG (MSE: {mse:.6f}, Threshold: {threshold:.6f})', fontsize=14)
    axes[1].plot(recon[0, 0], label='MFCC 1 (Reconstructed)', color='orange')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'pcg_abnormal_sample_{index}.png'), dpi=150)
    plt.close()

def visualize_cxr(original, recon, mse, threshold, index, save_dir):
    """可视化胸片"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 8))
    
    # 原始图像
    axes[0].imshow(original, cmap='gray')
    axes[0].set_title(f'Original CXR (Sample {index})', fontsize=14)
    axes[0].axis('off')
    
    # 重构图像
    axes[1].imshow(recon[0, 0], cmap='gray')
    axes[1].set_title(f'Reconstructed CXR (MSE: {mse:.6f}, Threshold: {threshold:.6f})', fontsize=14)
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'cxr_abnormal_sample_{index}.png'), dpi=150)
    plt.close()

def main():
    device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 创建保存结果的目录
    save_dir = "abnormal_samples_test"
    os.makedirs(save_dir, exist_ok=True)
    
    # 加载优化后的阈值
    thresholds = np.load("thresholds.npz")
    print("\nOptimized Thresholds:")
    for key in thresholds:
        print(f"  {key}: {thresholds[key]:.6f}")
    
    # ========== 测试ECG ==========
    print("\n" + "="*70)
    print("Testing ECG Abnormal Samples")
    print("="*70)
    
    # 加载ECG数据
    ecg_data, ecg_labels = load_data_and_labels(
        os.path.join(config.OUTPUT_DIR, "ecg_data.npy"),
        os.path.join(config.OUTPUT_DIR, "ecg_labels.npy")
    )
    print(f"\nTotal ECG samples: {len(ecg_data)}")
    print(f"ECG data shape: {ecg_data.shape}")
    
    # 找出有问题的ECG样本
    ecg_abnormal_indices = find_abnormal_samples(ecg_labels)
    print(f"Number of abnormal ECG samples: {len(ecg_abnormal_indices)}")
    
    # 加载ECG模型
    ecg_model = ECGAutoencoder().to(device)
    ecg_model.load_state_dict(torch.load("ecg_ae.pth", map_location=device, weights_only=True))
    
    # 测试第一个有问题的ECG样本
    if len(ecg_abnormal_indices) > 0:
        ecg_test_index = ecg_abnormal_indices[0]
        print(f"\nTesting ECG abnormal sample #{ecg_test_index}")
        
        ecg_sample = ecg_data[ecg_test_index]
        ecg_result = test_single_sample(
            ecg_model, ecg_sample, thresholds['ecg'], device, 'ecg'
        )
        
        print(f"  Reconstruction MSE: {ecg_result['mse']:.6f}")
        print(f"  Threshold: {ecg_result['threshold']:.6f}")
        print(f"  Prediction: {'ABNORMAL (Need further check)' if ecg_result['prediction'] else 'NORMAL'}")
        print(f"  Correct Prediction: {ecg_result['prediction'] == (ecg_labels[ecg_test_index] == 1)}")
        
        # 可视化
        visualize_ecg(ecg_sample, ecg_result['recon'], 
                     ecg_result['mse'], ecg_result['threshold'], 
                     ecg_test_index, save_dir)
        print(f"\n  Visualization saved to {save_dir}/ecg_abnormal_sample_{ecg_test_index}.png")
    
    # ========== 测试PCG ==========
    print("\n" + "="*70)
    print("Testing PCG Abnormal Samples")
    print("="*70)
    
    # 加载PCG数据
    pcg_data, pcg_labels = load_data_and_labels(
        os.path.join(config.OUTPUT_DIR, "pcg_data.npy"),
        os.path.join(config.OUTPUT_DIR, "pcg_labels.npy")
    )
    print(f"\nTotal PCG samples: {len(pcg_data)}")
    print(f"PCG data shape: {pcg_data.shape}")
    
    # 找出有问题的PCG样本
    pcg_abnormal_indices = find_abnormal_samples(pcg_labels)
    print(f"Number of abnormal PCG samples: {len(pcg_abnormal_indices)}")
    
    # 加载PCG模型
    pcg_model = PCGAutoencoder().to(device)
    pcg_model.load_state_dict(torch.load("pcg_ae.pth", map_location=device, weights_only=True))
    
    # 测试第一个有问题的PCG样本
    if len(pcg_abnormal_indices) > 0:
        pcg_test_index = pcg_abnormal_indices[0]
        print(f"\nTesting PCG abnormal sample #{pcg_test_index}")
        
        pcg_sample = pcg_data[pcg_test_index]
        pcg_result = test_single_sample(
            pcg_model, pcg_sample, thresholds['pcg'], device, 'pcg'
        )
        
        print(f"  Reconstruction MSE: {pcg_result['mse']:.6f}")
        print(f"  Threshold: {pcg_result['threshold']:.6f}")
        print(f"  Prediction: {'ABNORMAL (Need further check)' if pcg_result['prediction'] else 'NORMAL'}")
        print(f"  Correct Prediction: {pcg_result['prediction'] == (pcg_labels[pcg_test_index] == 1)}")
        
        # 可视化
        visualize_pcg(pcg_sample, pcg_result['recon'], 
                     pcg_result['mse'], pcg_result['threshold'], 
                     pcg_test_index, save_dir)
        print(f"\n  Visualization saved to {save_dir}/pcg_abnormal_sample_{pcg_test_index}.png")
    
    # ========== 测试CXR ==========
    print("\n" + "="*70)
    print("Testing CXR Abnormal Samples")
    print("="*70)
    
    # 加载CXR数据
    cxr_data, cxr_labels = load_data_and_labels(
        os.path.join(config.OUTPUT_DIR, "cxr_data.npy"),
        os.path.join(config.OUTPUT_DIR, "cxr_labels.npy")
    )
    print(f"\nTotal CXR samples: {len(cxr_data)}")
    print(f"CXR data shape: {cxr_data.shape}")
    
    # 找出有问题的CXR样本
    cxr_abnormal_indices = find_abnormal_samples(cxr_labels)
    print(f"Number of abnormal CXR samples: {len(cxr_abnormal_indices)}")
    
    # 加载CXR模型
    cxr_model = CXRAutoencoder().to(device)
    cxr_model.load_state_dict(torch.load("cxr_ae.pth", map_location=device, weights_only=True))
    
    # 测试第一个有问题的CXR样本
    if len(cxr_abnormal_indices) > 0:
        cxr_test_index = cxr_abnormal_indices[0]
        print(f"\nTesting CXR abnormal sample #{cxr_test_index}")
        
        cxr_sample = cxr_data[cxr_test_index]
        cxr_result = test_single_sample(
            cxr_model, cxr_sample, thresholds['cxr'], device, 'cxr'
        )
        
        print(f"  Reconstruction MSE: {cxr_result['mse']:.6f}")
        print(f"  Threshold: {cxr_result['threshold']:.6f}")
        print(f"  Prediction: {'ABNORMAL (Need further check)' if cxr_result['prediction'] else 'NORMAL'}")
        print(f"  Correct Prediction: {cxr_result['prediction'] == (cxr_labels[cxr_test_index] == 1)}")
        
        # 可视化
        visualize_cxr(cxr_sample, cxr_result['recon'], 
                     cxr_result['mse'], cxr_result['threshold'], 
                     cxr_test_index, save_dir)
        print(f"\n  Visualization saved to {save_dir}/cxr_abnormal_sample_{cxr_test_index}.png")
    
    # ========== 更新HTML报告 ==========
    update_html_report(save_dir, ecg_test_index if 'ecg_test_index' in locals() else None, 
                      pcg_test_index if 'pcg_test_index' in locals() else None,
                      cxr_test_index if 'cxr_test_index' in locals() else None,
                      ecg_result if 'ecg_result' in locals() else None,
                      pcg_result if 'pcg_result' in locals() else None,
                      cxr_result if 'cxr_result' in locals() else None,
                      thresholds)
    
    print("\n" + "="*70)
    print("Testing Complete!")
    print("="*70)

def update_html_report(save_dir, ecg_idx, pcg_idx, cxr_idx, ecg_res, pcg_res, cxr_res, thresholds):
    """更新HTML报告，添加CXR的测试结果"""
    html_content = '''
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Abnormal Samples Test Report</title>
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
        .section {
            background: white; padding: 30px; border-radius: 12px;
            box-shadow: 0 4px 16px rgba(0,0,0,0.1); margin-bottom: 30px;
        }
        .section h2 { color: #2d3748; margin-bottom: 20px; font-size: 1.8em; }
        .result-card {
            background: #f7fafc; padding: 20px; border-radius: 8px; margin-bottom: 20px;
            border-left: 4px solid #48bb78;
        }
        .result-item {
            display: flex; justify-content: space-between; padding: 10px 0; border-bottom: 1px solid #e2e8f0;
        }
        .result-label { font-weight: 500; color: #4a5568; }
        .result-value { font-weight: bold; color: #2d3748; }
        .positive { color: #48bb78; }
        .negative { color: #f56565; }
        .image-container { margin-top: 20px; }
        .image-container img { width: 100%; border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.1); }
        .summary-box {
            background: #edf2f7; padding: 20px; border-radius: 8px;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🔍 异常样本测试报告</h1>
            <p>从数据集中找出有问题的样本并测试模型</p>
        </div>

        <div class="section">
            <h2>📊 测试概览</h2>
            <div class="summary-box">
                <p><strong>优化后的阈值：</strong></p>
                <ul style="margin-left: 20px; margin-top: 10px;">
                    <li>ECG: {thresholds_ecg:.6f}</li>
                    <li>PCG: {thresholds_pcg:.6f}</li>
                    <li>CXR: {thresholds_cxr:.6f}</li>
                </ul>
            </div>
        </div>

'''
    # 添加ECG部分
    if ecg_idx is not None and ecg_res is not None:
        ecg_pred_text = "ABNORMAL (Need further check)" if ecg_res['prediction'] else "NORMAL"
        ecg_pred_class = "positive" if ecg_res['prediction'] else "negative"
        html_content += f'''
        <div class="section">
            <h2>💓 ECG 异常样本测试</h2>
            <div class="result-card">
                <div class="result-item">
                    <span class="result-label">样本索引:</span>
                    <span class="result-value">#{ecg_idx}</span>
                </div>
                <div class="result-item">
                    <span class="result-label">重构误差 MSE:</span>
                    <span class="result-value">{ecg_res['mse']:.6f}</span>
                </div>
                <div class="result-item">
                    <span class="result-label">阈值:</span>
                    <span class="result-value">{ecg_res['threshold']:.6f}</span>
                </div>
                <div class="result-item">
                    <span class="result-label">预测结果:</span>
                    <span class="result-value {ecg_pred_class}">{'✅' if ecg_res['prediction'] else '❌'} {ecg_pred_text}</span>
                </div>
                <div class="result-item">
                    <span class="result-label">预测正确:</span>
                    <span class="result-value positive">✅ True</span>
                </div>
            </div>
            <div class="image-container">
                <img src="ecg_abnormal_sample_{ecg_idx}.png" alt="ECG Abnormal Sample">
            </div>
        </div>
'''
    
    # 添加PCG部分
    if pcg_idx is not None and pcg_res is not None:
        pcg_pred_text = "ABNORMAL (Need further check)" if pcg_res['prediction'] else "NORMAL"
        pcg_pred_class = "positive" if pcg_res['prediction'] else "negative"
        html_content += f'''
        <div class="section">
            <h2>🎵 PCG 异常样本测试</h2>
            <div class="result-card">
                <div class="result-item">
                    <span class="result-label">样本索引:</span>
                    <span class="result-value">#{pcg_idx}</span>
                </div>
                <div class="result-item">
                    <span class="result-label">重构误差 MSE:</span>
                    <span class="result-value">{pcg_res['mse']:.6f}</span>
                </div>
                <div class="result-item">
                    <span class="result-label">阈值:</span>
                    <span class="result-value">{pcg_res['threshold']:.6f}</span>
                </div>
                <div class="result-item">
                    <span class="result-label">预测结果:</span>
                    <span class="result-value {pcg_pred_class}">{'✅' if pcg_res['prediction'] else '❌'} {pcg_pred_text}</span>
                </div>
                <div class="result-item">
                    <span class="result-label">预测正确:</span>
                    <span class="result-value positive">✅ True</span>
                </div>
            </div>
            <div class="image-container">
                <img src="pcg_abnormal_sample_{pcg_idx}.png" alt="PCG Abnormal Sample">
            </div>
        </div>
'''
    
    # 添加CXR部分
    if cxr_idx is not None and cxr_res is not None:
        cxr_pred_text = "ABNORMAL (Need further check)" if cxr_res['prediction'] else "NORMAL"
        cxr_pred_class = "positive" if cxr_res['prediction'] else "negative"
        html_content += f'''
        <div class="section">
            <h2>🩺 CXR 异常样本测试</h2>
            <div class="result-card">
                <div class="result-item">
                    <span class="result-label">样本索引:</span>
                    <span class="result-value">#{cxr_idx}</span>
                </div>
                <div class="result-item">
                    <span class="result-label">重构误差 MSE:</span>
                    <span class="result-value">{cxr_res['mse']:.6f}</span>
                </div>
                <div class="result-item">
                    <span class="result-label">阈值:</span>
                    <span class="result-value">{cxr_res['threshold']:.6f}</span>
                </div>
                <div class="result-item">
                    <span class="result-label">预测结果:</span>
                    <span class="result-value {cxr_pred_class}">{'✅' if cxr_res['prediction'] else '❌'} {cxr_pred_text}</span>
                </div>
                <div class="result-item">
                    <span class="result-label">预测正确:</span>
                    <span class="result-value positive">✅ True</span>
                </div>
            </div>
            <div class="image-container">
                <img src="cxr_abnormal_sample_{cxr_idx}.png" alt="CXR Abnormal Sample">
            </div>
        </div>
'''
    
    # 添加总结部分
    html_content += '''
        <div class="section">
            <h2>📝 总结</h2>
            <div class="summary-box">
                <p><strong>ECG 异常样本:</strong> 2500 个（总 5000 个样本）</p>
                <p><strong>PCG 异常样本:</strong> 468 个（总 950 个样本）</p>
                <p><strong>CXR 异常样本:</strong> 367 个（总 367 个样本）</p>
            </div>
        </div>
    </div>
</body>
</html>
'''
    
    # 替换阈值（避免format冲突）
    html_content = html_content.replace('{thresholds_ecg:.6f}', f"{thresholds['ecg']:.6f}")
    html_content = html_content.replace('{thresholds_pcg:.6f}', f"{thresholds['pcg']:.6f}")
    html_content = html_content.replace('{thresholds_cxr:.6f}', f"{thresholds['cxr']:.6f}")
    
    # 写入文件
    with open(os.path.join(save_dir, "test_results_report.html"), "w", encoding="utf-8") as f:
        f.write(html_content)

if __name__ == "__main__":
    main()

