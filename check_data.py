import os
import pandas as pd

# 检查目录结构
print("当前 data 目录内容：")
data_dir = "d:/chd_anomaly_detection/data"
for item in os.listdir(data_dir):
    print(f"  - {item}")

# 检查 Circor 的 training_data.csv
print("\n检查 Circor 数据集：")
circor_csv = os.path.join(data_dir, "circor", "training_data.csv")
if os.path.exists(circor_csv):
    df_circor = pd.read_csv(circor_csv)
    print(f"  列名：{list(df_circor.columns)}")
    print(f"  前5行：")
    print(df_circor.head())

# 检查 PTB-XL 相关内容（如果有）
print("\n检查 PTB-XL 相关文件：")
ptbxl_dir = os.path.join(data_dir, "ptbxl")
if os.path.exists(ptbxl_dir):
    print(f"  ptbxl 目录内容：{os.listdir(ptbxl_dir)}")
    # 查找 CSV 文件
    csv_files = [f for f in os.listdir(ptbxl_dir) if f.endswith('.csv')]
    if csv_files:
        csv_file = os.path.join(ptbxl_dir, csv_files[0])
        df_ptbxl = pd.read_csv(csv_file)
        print(f"  找到 CSV 文件：{csv_files[0]}")
        print(f"  列名：{list(df_ptbxl.columns)}")
        print(f"  前5行：")
        print(df_ptbxl.head())
else:
    print("  ptbxl 目录不存在！")
