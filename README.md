# 先天性心脏病多模态异常检测

这是一个基于自编码器的先天性心脏病（CHD）异常筛查原型系统，支持三类模态：

- ECG：心电图信号
- PCG：心音音频
- CXR：胸片图像

系统使用正常样本训练自编码器，通过重构误差（MSE）与阈值比较来判断样本是否存在异常。当前项目适合科研原型、课程展示和工程演示，不应作为临床诊断依据。

## 云端部署

仓库已包含 Render 部署配置：

- `render.yaml`
- `requirements-deploy.txt`
- `DEPLOY.md`

部署步骤和 curl 测试命令见 [DEPLOY.md](DEPLOY.md)。

## 项目结构

```text
chd_anomaly_detection/
├── config.py                  # 路径与训练参数配置
├── data_preprocessing.py      # 数据预处理
├── model.py                   # ECG/PCG/CXR 自编码器模型
├── train.py                   # 训练脚本
├── inference.py               # 命令行推理
├── app.py                     # Flask API 服务
├── export_onnx.py             # ONNX 导出
├── requirements.txt           # Python 依赖
├── thresholds.npz             # 三个模态的异常阈值
├── *_ae.pth                   # 训练后的模型权重
└── evaluation_results/        # 评估图表与报告
```

## 安装依赖

建议使用虚拟环境：

```bash
pip install -r requirements.txt
```

## 数据配置

默认数据路径在 `config.py` 中配置。也可以通过环境变量覆盖：

```bash
set CHD_DATA_ROOT=D:\chd_anomaly_detection\data
set PTBXL_PATH=D:\chd_anomaly_detection\data\ptbxl
set CIRCOR_PATH=D:\chd_anomaly_detection\data\circor
set CHD_CXR_PATH=D:\chd_anomaly_detection\data\chd_xray
```

## 快速开始

### 1. 数据预处理

```bash
python data_preprocessing.py
```

预处理后会在 `data/` 目录生成：

- `normal_ecg.npy`
- `normal_pcg.npy`
- `normal_cxr.npy`

### 2. 训练模型

```bash
python train.py
```

训练完成后会生成：

- `ecg_ae.pth`
- `pcg_ae.pth`
- `cxr_ae.pth`
- `thresholds.npz`

### 3. 命令行推理

```bash
python inference.py --ecg path\to\ecg_record_without_extension
python inference.py --pcg path\to\heart_sound.wav
python inference.py --cxr path\to\chest_xray.jpg
```

可以同时传入多个模态：

```bash
python inference.py --ecg path\to\record --pcg path\to\audio.wav --cxr path\to\xray.jpg
```

### 4. 启动 API 服务

```bash
python app.py
```

服务默认启动在：

```text
http://0.0.0.0:7860
```

API：

- `GET /`：服务信息
- `GET /api/health`：健康检查
- `POST /api/predict/ecg`：ECG 推理
- `POST /api/predict/pcg`：PCG 推理
- `POST /api/predict/cxr`：CXR 推理
- `GET /api/fusion/config`：多模态融合配置与权重说明
- `POST /api/predict/multimodal`：多模态加权 late-fusion 推理

## API 上传说明

PCG 与 CXR 使用 `multipart/form-data` 上传字段 `file`。

ECG 支持两种方式：

- 上传 WFDB 文件对：字段 `hea` 上传 `.hea` 文件，字段 `dat` 上传 `.dat` 文件
- 传入本机记录路径：JSON 或表单字段 `record_path`，路径不包含扩展名

多模态接口可在同一个请求中上传：

- `ecg_hea`
- `ecg_dat`
- `pcg_file`
- `cxr_file`

## 多模态融合策略

当前 API 已不再只是并列返回三个单模态结果，而是使用加权 late fusion 生成综合风险：

```text
单模态标准化分数 = reconstruction_mse / modality_threshold
融合分数 = sum(归一化权重 * 单模态标准化分数)
```

默认权重：

| 模态 | 权重 | 依据 |
| --- | ---: | --- |
| ECG | 0.50 | 当前 ROC AUC 最高，信号最稳定 |
| PCG | 0.30 | 有一定区分度，但弱于 ECG |
| CXR | 0.20 | 当前 CXR AUC 较弱，因此降低权重 |

判定规则：

- `fusion_score < 1.0`：low
- `1.0 <= fusion_score < 1.5`：elevated
- `fusion_score >= 1.5`：high

缺失模态处理：如果请求中只提供 ECG/PCG/CXR 的一部分，系统会只使用已提供模态，并将这些模态的权重重新归一化。例如只提供 ECG 和 PCG 时，CXR 权重会被移除，ECG 与 PCG 权重按比例重新归一化。

融合配置可通过接口查看：

```bash
curl http://127.0.0.1:7860/api/fusion/config
```

## 当前评估结果

`test_results_complete/evaluation_summary.txt` 中记录的评估摘要：

| 模态 | Accuracy | Recall/Sensitivity | Specificity | ROC AUC |
| --- | ---: | ---: | ---: | ---: |
| ECG | 0.6000 | 0.9000 | 0.3000 | 0.7271 |
| PCG | 0.5389 | 0.9017 | 0.1867 | 0.6062 |
| CXR | 0.4471 | 0.8029 | 0.0913 | 0.4683 |

当前模型更适合作为高召回筛查原型。CXR 单模态表现较弱，实际展示或报告中建议降低其权重，或后续替换为监督分类/预训练视觉模型方案。

## 重要声明

本项目仅用于研究、教学和算法演示。输出结果不能替代医生诊断，也不能作为医疗决策依据。如检测结果提示异常或用户存在不适，应以正规医疗机构检查为准。
