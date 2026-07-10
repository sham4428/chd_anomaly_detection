import numpy as np
import os

# 新的临床优化阈值
new_thresholds = {
    "ecg": 0.12,
    "pcg": 0.08,
    "cxr": 0.05
}

# 保存新阈值
np.savez("thresholds.npz", **new_thresholds)
print("New thresholds saved:")
for key, value in new_thresholds.items():
    print(f"  {key}: {value:.4f}")

# 打印当前数据文件信息
print("\nData files info:")
data_dir = "data"
for f in os.listdir(data_dir):
    if f.endswith(".npy"):
        data = np.load(os.path.join(data_dir, f))
        print(f"  {f}: shape {data.shape}")
