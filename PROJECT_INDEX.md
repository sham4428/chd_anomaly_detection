# 项目分级索引

这个索引只回答两件事：先看什么，以及每个文件属于哪一层。

## 阅读顺序

第一次理解项目时只按下面顺序阅读：

```text
config.py
  -> data_preprocessing.py
  -> model.py
  -> train.py
  -> optimize_thresholds.py
  -> app.py
  -> streamlit_app.py
```

`evaluate_model.py` 用于科研评估，`inference.py` 用于本地命令行推理，它们不是训练主流程的一部分。

## L1 核心算法

这些文件决定数据、模型、训练和阈值。

| 文件 | 作用 | 主要输出 |
| --- | --- | --- |
| `config.py` | 路径、batch size、epoch、学习率和设备配置 | 运行参数 |
| `data_preprocessing.py` | 将 ECG、PCG、CXR 原始数据处理成统一数组 | `data/*.npy` |
| `model.py` | 三个自编码器及新旧权重兼容加载 | PyTorch 模型 |
| `train.py` | 每个模态使用自己的全部正常样本独立训练 | `*_ae.pth` |
| `optimize_thresholds.py` | 在 calibration set 选阈值，在 held-out test 报告 | `thresholds.npz` |

## L2 推理与评估

这些文件消费模型权重，不定义训练数据。

| 文件 | 作用 |
| --- | --- |
| `app.py` | Flask API、输入校验、单模态预测和 late fusion |
| `streamlit_app.py` | 上传文件并调用 Flask API 的网页前端 |
| `inference.py` | 不启动网页时使用的命令行推理入口 |
| `evaluate_model.py` | 单模态指标、ROC、误差方向和分布图 |
| `export_onnx.py` | 将兼容加载后的模型导出为 ONNX |
| `check_data.py` | 快速查看预处理数据数量和形状 |

## L3 文档与部署

| 文件或目录 | 作用 |
| --- | --- |
| `README.md` | GitHub 首页、启动方式、API 文档和效果截图 |
| `PROJECT_GUIDE.md` | 较详细的项目理解说明 |
| `PROJECT_INDEX.md` | 当前分级索引 |
| `DEPLOY.md` | Render 和 EC2 部署步骤 |
| `render.yaml` | Render API 服务配置 |
| `requirements.txt` | 本地完整依赖，包括训练和 Streamlit |
| `requirements-deploy.txt` | 云端 Flask API 的精简依赖 |
| `prepare_upload.py` | 在项目外创建不含数据的安全上传副本 |
| `docs/architecture.drawio` | 可编辑的 draw.io 架构源文件 |
| `docs/architecture.svg` | README 展示的架构图 |
| `docs/streamlit_frontend.png` | README 展示的前端截图 |

## L4 数据与生成物

这些内容通常体积较大，或者可以由代码重新生成。

| 路径 | 类型 | 说明 |
| --- | --- | --- |
| `data/ptbxl/` | 原始数据 | ECG 数据源 |
| `data/circor/` | 原始数据 | PCG 数据源 |
| `data/chd_xray/` | 原始数据 | CXR 数据源 |
| `data/*.npy` | 预处理结果 | 训练或评估数组 |
| `ecg_ae.pth` | 模型权重 | API 当前默认 ECG 权重 |
| `pcg_ae.pth` | 模型权重 | API 当前默认 PCG 权重 |
| `cxr_ae.pth` | 模型权重 | API 当前默认 CXR 权重 |
| `thresholds.npz` | 阈值 | 三个模态的异常阈值 |
| `*_best.pth` / `*_final.pth` | 训练生成物 | 中间或历史 checkpoint |
| `abnormal_samples_test/` | 本地测试数据 | 手工异常样本，不进入训练主流程 |

## L5 历史代码与缓存

| 路径 | 处理规则 |
| --- | --- |
| `legacy/` | 仅供追溯旧实验，不从主流程调用 |
| `.idea/` | IDE 本地配置，不属于项目代码 |
| `__pycache__/` | Python 缓存，可随时重新生成 |

## 权重版本

当前仓库自带的是历史结构权重。`model.py` 会自动识别，避免架构不匹配导致服务崩溃：

```text
ECG -> ecg_legacy_no_bottleneck
PCG -> pcg_legacy_no_bottleneck
CXR -> cxr_legacy_patch_v1
```

重新执行 `train.py` 后，新模型才会使用 bottleneck 和 CXR CNN decoder。训练完成后必须重新执行 `optimize_thresholds.py`，不能继续沿用旧阈值。

## 维护规则

1. 主流程只允许从 L1 指向 L1/L2，不从 `legacy/` 导入代码。
2. 模型结构、权重和阈值必须作为同一版本更新。
3. `data/`、模型权重和评估报告不与源码混合移动。
4. 上传副本放在项目目录外，默认不包含 `data/`。
5. 删除目录前先确认它是缓存、历史输出还是唯一数据源。
