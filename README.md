# 先天性心脏病多模态异常检测

基于自编码器的先天性心脏病异常检测系统，支持心电图(ECG)、心音(PCG)和胸片(CXR)三种模态。

## 项目结构

```
chd_anomaly_detection/
├── config.py              # 配置文件
├── data_preprocessing.py  # 数据预处理脚本
├── model.py              # 模型定义
├── train.py             # 训练脚本
├── inference.py         # 推理脚本
├── export_onnx.py      # ONNX导出脚本
├── app.py              # Flask服务
└── requirements.txt    # 依赖包
```

## 安装依赖

```bash
pip install -r requirements.txt
```

## 快速开始

### 1. 数据预处理

首先确保数据集路径在 `config.py` 中配置正确，然后运行：

```bash
python data_preprocessing.py
```

### 2. 模型训练

```bash
python train.py
```

训练完成后会生成：
- `ecg_ae.pth`
- `pcg_ae.pth`
- `cxr_ae.pth`
- `thresholds.npz`

### 3. 推理使用

#### 命令行推理

```bash
# 推理ECG
python inference.py --ecg path/to/ecg_file

# 推理PCG
python inference.py --pcg path/to/audio.wav

# 推理CXR
python inference.py --cxr path/to/image.jpg
```

### 4. 导出ONNX模型

```bash
python export_onnx.py
```

### 5. 启动Flask服务

```bash
python app.py
```

服务将在 `http://0.0.0.0:7860` 启动

#### API接口

- `GET /` - 服务信息
- `GET /api/health` - 健康检查
- `POST /api/predict/pcg` - PCG预测
- `POST /api/predict/cxr` - CXR预测

## 推理原理

1. 使用正常数据训练自编码器
2. 计算正常样本重构误差的95%分位数作为阈值
3. 对于新样本，计算其重构误差(MSE)
4. 如果误差 > 阈值 → 判定为异常，建议就医
