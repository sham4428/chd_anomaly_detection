# 数据集路径配置（请修改为你本地的实际路径）
PTBXL_PATH = "D:\chd_anomaly_detection\data\ptbxl"                    # PTB-XL根目录
CIRCOR_PATH = "D:\chd_anomaly_detection\data\circor"        # Circor心音数据集根目录
CHD_CXR_PATH = "D:\chd_anomaly_detection\data\chd_xray" # Kaggle CHD胸片数据集根目录

# 预处理输出目录
OUTPUT_DIR = "./data"

# 训练参数
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 1e-4
DEVICE = "cuda"  # 或 "cpu"