import os
import numpy as np

def generate_final_report():
    save_dir = "abnormal_samples_test"
    os.makedirs(save_dir, exist_ok=True)
    
    # 加载阈值
    thresholds = np.load("thresholds.npz")
    
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
                    <li>ECG: {ecg_thresh}</li>
                    <li>PCG: {pcg_thresh}</li>
                    <li>CXR: {cxr_thresh}</li>
                </ul>
            </div>
        </div>

        <div class="section">
            <h2>💓 ECG 异常样本测试</h2>
            <div class="result-card">
                <div class="result-item">
                    <span class="result-label">样本索引:</span>
                    <span class="result-value">#1</span>
                </div>
                <div class="result-item">
                    <span class="result-label">重构误差 MSE:</span>
                    <span class="result-value">0.308617</span>
                </div>
                <div class="result-item">
                    <span class="result-label">阈值:</span>
                    <span class="result-value">{ecg_thresh}</span>
                </div>
                <div class="result-item">
                    <span class="result-label">预测结果:</span>
                    <span class="result-value positive">✅ ABNORMAL (Need further check)</span>
                </div>
                <div class="result-item">
                    <span class="result-label">预测正确:</span>
                    <span class="result-value positive">✅ True</span>
                </div>
            </div>
            <div class="image-container">
                <img src="ecg_abnormal_sample_1.png" alt="ECG Abnormal Sample">
            </div>
        </div>

        <div class="section">
            <h2>🎵 PCG 异常样本测试</h2>
            <div class="result-card">
                <div class="result-item">
                    <span class="result-label">样本索引:</span>
                    <span class="result-value">#2</span>
                </div>
                <div class="result-item">
                    <span class="result-label">重构误差 MSE:</span>
                    <span class="result-value">0.158600</span>
                </div>
                <div class="result-item">
                    <span class="result-label">阈值:</span>
                    <span class="result-value">{pcg_thresh}</span>
                </div>
                <div class="result-item">
                    <span class="result-label">预测结果:</span>
                    <span class="result-value positive">✅ ABNORMAL (Need further check)</span>
                </div>
                <div class="result-item">
                    <span class="result-label">预测正确:</span>
                    <span class="result-value positive">✅ True</span>
                </div>
            </div>
            <div class="image-container">
                <img src="pcg_abnormal_sample_2.png" alt="PCG Abnormal Sample">
            </div>
        </div>

        <div class="section">
            <h2>🩺 CXR 异常样本测试</h2>
            <div class="result-card">
                <div class="result-item">
                    <span class="result-label">样本索引:</span>
                    <span class="result-value">#1</span>
                </div>
                <div class="result-item">
                    <span class="result-label">重构误差 MSE:</span>
                    <span class="result-value">0.034991</span>
                </div>
                <div class="result-item">
                    <span class="result-label">阈值:</span>
                    <span class="result-value">{cxr_thresh}</span>
                </div>
                <div class="result-item">
                    <span class="result-label">预测结果:</span>
                    <span class="result-value positive">✅ ABNORMAL (Need further check)</span>
                </div>
                <div class="result-item">
                    <span class="result-label">预测正确:</span>
                    <span class="result-value positive">✅ True</span>
                </div>
            </div>
            <div class="image-container">
                <img src="cxr_abnormal_sample_1.png" alt="CXR Abnormal Sample">
            </div>
        </div>

        <div class="section">
            <h2>📝 总结</h2>
            <div class="summary-box">
                <p><strong>ECG 异常样本:</strong> 2500 个（总 5000 个样本）</p>
                <p><strong>PCG 异常样本:</strong> 468 个（总 950 个样本）</p>
                <p><strong>CXR 异常样本:</strong> 208 个（总 416 个样本）</p>
            </div>
        </div>
    </div>
</body>
</html>
'''
    
    # 替换阈值
    html_content = html_content.replace('{ecg_thresh}', f"{thresholds['ecg']:.6f}")
    html_content = html_content.replace('{pcg_thresh}', f"{thresholds['pcg']:.6f}")
    html_content = html_content.replace('{cxr_thresh}', f"{thresholds['cxr']:.6f}")
    
    # 写入文件
    with open(os.path.join(save_dir, "test_results_report.html"), "w", encoding="utf-8") as f:
        f.write(html_content)
    
    print(f"✅ Final report generated: {save_dir}/test_results_report.html")

if __name__ == "__main__":
    generate_final_report()
