# 云端部署说明

本项目是 Flask API，可以部署到 Render 或 AWS EC2。推荐先用 Render，因为它可以直接连接 GitHub 仓库并自动生成公网 HTTPS 链接。

## 方案 A：Render 部署

仓库里已经包含：

- `render.yaml`：Render Blueprint 配置
- `requirements-deploy.txt`：云端 API 运行依赖

### 使用 Blueprint 部署

1. 登录 Render。
2. 选择 `New` -> `Blueprint`。
3. 连接 GitHub 仓库：`sham4428/chd_anomaly_detection`。
4. Render 会读取 `render.yaml`。
5. 确认服务名为 `chd-anomaly-detection`。
6. 创建服务并等待构建完成。

构建完成后，Render 会生成类似下面的公网地址：

```text
https://chd-anomaly-detection.onrender.com
```

如果这个名字已被占用，Render 会生成带后缀的地址，以控制台显示为准。

### curl 测试

健康检查：

```bash
curl https://chd-anomaly-detection.onrender.com/api/health
```

融合配置：

```bash
curl https://chd-anomaly-detection.onrender.com/api/fusion/config
```

CXR 图片预测示例：

```bash
curl -X POST https://chd-anomaly-detection.onrender.com/api/predict/cxr \
  -F "file=@sample_xray.jpg"
```

PCG 音频预测示例：

```bash
curl -X POST https://chd-anomaly-detection.onrender.com/api/predict/pcg \
  -F "file=@sample_heart_sound.wav"
```

多模态预测示例：

```bash
curl -X POST https://chd-anomaly-detection.onrender.com/api/predict/multimodal \
  -F "pcg_file=@sample_heart_sound.wav" \
  -F "cxr_file=@sample_xray.jpg"
```

## 注意事项

- Render Free 实例资源有限。项目依赖 PyTorch、torchvision 和 librosa，首次构建可能较慢，也可能因为内存不足失败。
- 如果 Free 实例失败，优先升级到更高内存实例，或改造成 ONNX Runtime 版本以降低内存占用。
- 模型权重文件必须存在于仓库根目录：`ecg_ae.pth`、`pcg_ae.pth`、`cxr_ae.pth`、`thresholds.npz`。
- 本项目仅用于研究、教学和演示，不能替代医生诊断。

## 方案 B：AWS EC2 免费额度

1. 创建 Ubuntu EC2 实例。
2. 安装 Python、Git、Nginx。
3. 克隆仓库：

```bash
git clone https://github.com/sham4428/chd_anomaly_detection.git
cd chd_anomaly_detection
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-deploy.txt
gunicorn app:app --bind 0.0.0.0:7860 --workers 1 --threads 2 --timeout 180
```

4. 在安全组开放 7860 或配置 Nginx 反向代理到 80/443。

EC2 更灵活，但需要手动维护系统、端口、安全组和进程守护。
