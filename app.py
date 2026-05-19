import os
import io
import tempfile
import numpy as np
import torch
import wfdb
import librosa
from PIL import Image
from torchvision import transforms
from flask import Flask, request, jsonify
from flask_cors import CORS
from model import ECGAutoencoder, PCGAutoencoder, CXRAutoencoder


app = Flask(__name__)
CORS(app)


class CHDDetector:
    def __init__(self, model_dir="./"):
        self.device = torch.device('cpu')
        self._load_models(model_dir)
        self._load_thresholds(model_dir)

    def _load_models(self, model_dir):
        self.ecg_model = ECGAutoencoder().to(self.device)
        self.ecg_model.load_state_dict(torch.load(os.path.join(model_dir, "ecg_ae.pth"), map_location='cpu', weights_only=True))
        self.ecg_model.eval()

        self.pcg_model = PCGAutoencoder().to(self.device)
        self.pcg_model.load_state_dict(torch.load(os.path.join(model_dir, "pcg_ae.pth"), map_location='cpu', weights_only=True))
        self.pcg_model.eval()

        self.cxr_model = CXRAutoencoder().to(self.device)
        self.cxr_model.load_state_dict(torch.load(os.path.join(model_dir, "cxr_ae.pth"), map_location='cpu', weights_only=True))
        self.cxr_model.eval()

    def _load_thresholds(self, model_dir):
        thresh_data = np.load(os.path.join(model_dir, "thresholds.npz"))
        self.ecg_thresh = float(thresh_data["ecg"])
        self.pcg_thresh = float(thresh_data["pcg"])
        self.cxr_thresh = float(thresh_data["cxr"])

    def predict_ecg(self, ecg_data):
        x = torch.tensor(ecg_data, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            recon = self.ecg_model(x)
            mse = torch.mean((recon - x) ** 2, dim=(1, 2)).item()
        is_abnormal = mse > self.ecg_thresh
        return {
            "modality": "ECG",
            "mse": mse,
            "threshold": self.ecg_thresh,
            "is_abnormal": is_abnormal,
            "recommendation": "检测到异常，建议就医" if is_abnormal else "心电图正常"
        }

    def predict_pcg(self, mfcc_data):
        x = torch.tensor(mfcc_data, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            recon = self.pcg_model(x)
            mse = torch.mean((recon - x) ** 2, dim=(1, 2)).item()
        is_abnormal = mse > self.pcg_thresh
        return {
            "modality": "PCG",
            "mse": mse,
            "threshold": self.pcg_thresh,
            "is_abnormal": is_abnormal,
            "recommendation": "检测到异常，建议就医" if is_abnormal else "心音正常"
        }

    def predict_cxr(self, img_data):
        x = img_data.unsqueeze(0)
        with torch.no_grad():
            recon = self.cxr_model(x)
            mse = torch.mean((recon - x) ** 2, dim=(1, 2, 3)).item()
        is_abnormal = mse > self.cxr_thresh
        return {
            "modality": "CXR",
            "mse": mse,
            "threshold": self.cxr_thresh,
            "is_abnormal": is_abnormal,
            "recommendation": "检测到异常，建议就医" if is_abnormal else "胸片正常"
        }


detector = CHDDetector()


@app.route('/')
def index():
    return jsonify({
        "service": "先天性心脏病多模态异常检测",
        "version": "1.0",
        "endpoints": {
            "/api/health": "健康检查",
            "/api/predict/ecg": "ECG预测",
            "/api/predict/pcg": "PCG预测",
            "/api/predict/cxr": "CXR预测"
        }
    })


@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({"status": "ok", "message": "服务运行正常"})


@app.route('/api/predict/pcg', methods=['POST'])
def predict_pcg():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "未上传文件"}), 400

        file = request.files['file']
        y, sr = librosa.load(io.BytesIO(file.read()), sr=2000)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, n_fft=512, hop_length=256)
        if mfcc.shape[1] < 200:
            mfcc = np.pad(mfcc, ((0, 0), (0, 200 - mfcc.shape[1])), mode='constant')
        else:
            mfcc = mfcc[:, :200]
        mfcc = (mfcc - mfcc.mean()) / (mfcc.std() + 1e-8)
        result = detector.predict_pcg(mfcc)
        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/predict/cxr', methods=['POST'])
def predict_cxr():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "未上传文件"}), 400

        file = request.files['file']
        img = Image.open(io.BytesIO(file.read())).convert("L")

        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        img_tensor = transform(img)
        result = detector.predict_cxr(img_tensor)
        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=7860, debug=False)
