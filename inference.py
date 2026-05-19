import os
import numpy as np
import torch
import wfdb
import librosa
from PIL import Image
from torchvision import transforms
from model import ECGAutoencoder, PCGAutoencoder, CXRAutoencoder
import config


class CHDDetector:
    def __init__(self, model_dir="./"):
        self.device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")
        self._load_models(model_dir)
        self._load_thresholds(model_dir)

    def _load_models(self, model_dir):
        self.ecg_model = ECGAutoencoder().to(self.device)
        self.ecg_model.load_state_dict(torch.load(os.path.join(model_dir, "ecg_ae.pth"), map_location=self.device, weights_only=True))
        self.ecg_model.eval()

        self.pcg_model = PCGAutoencoder().to(self.device)
        self.pcg_model.load_state_dict(torch.load(os.path.join(model_dir, "pcg_ae.pth"), map_location=self.device, weights_only=True))
        self.pcg_model.eval()

        self.cxr_model = CXRAutoencoder().to(self.device)
        self.cxr_model.load_state_dict(torch.load(os.path.join(model_dir, "cxr_ae.pth"), map_location=self.device, weights_only=True))
        self.cxr_model.eval()

    def _load_thresholds(self, model_dir):
        thresh_data = np.load(os.path.join(model_dir, "thresholds.npz"))
        self.ecg_thresh = float(thresh_data["ecg"])
        self.pcg_thresh = float(thresh_data["pcg"])
        self.cxr_thresh = float(thresh_data["cxr"])

    def _preprocess_ecg(self, ecg_path):
        sig, _ = wfdb.rdsamp(ecg_path)
        sig = sig[:1000, :12].T
        sig = (sig - sig.mean(axis=1, keepdims=True)) / (sig.std(axis=1, keepdims=True) + 1e-8)
        return torch.tensor(sig, dtype=torch.float32).unsqueeze(0)

    def _preprocess_pcg(self, wav_path):
        y, sr = librosa.load(wav_path, sr=2000)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, n_fft=512, hop_length=256)
        if mfcc.shape[1] < 200:
            mfcc = np.pad(mfcc, ((0, 0), (0, 200 - mfcc.shape[1])), mode='constant')
        else:
            mfcc = mfcc[:, :200]
        mfcc = (mfcc - mfcc.mean()) / (mfcc.std() + 1e-8)
        return torch.tensor(mfcc, dtype=torch.float32).unsqueeze(0)

    def _preprocess_cxr(self, img_path):
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        img = Image.open(img_path).convert("L")
        img_tensor = transform(img).unsqueeze(0)
        return img_tensor

    def predict_ecg(self, ecg_path):
        x = self._preprocess_ecg(ecg_path).to(self.device)
        with torch.no_grad():
            recon = self.ecg_model(x)
            mse = torch.mean((recon - x) ** 2, dim=(1, 2)).item()
        is_abnormal = mse > self.ecg_thresh
        return {
            "modality": "ECG",
            "mse": mse,
            "threshold": self.ecg_thresh,
            "is_abnormal": is_abnormal,
            "recommendation": "⚠️ 检测到异常，建议就医" if is_abnormal else "✅ 心电图正常"
        }

    def predict_pcg(self, wav_path):
        x = self._preprocess_pcg(wav_path).to(self.device)
        with torch.no_grad():
            recon = self.pcg_model(x)
            mse = torch.mean((recon - x) ** 2, dim=(1, 2)).item()
        is_abnormal = mse > self.pcg_thresh
        return {
            "modality": "PCG",
            "mse": mse,
            "threshold": self.pcg_thresh,
            "is_abnormal": is_abnormal,
            "recommendation": "⚠️ 检测到异常，建议就医" if is_abnormal else "✅ 心音正常"
        }

    def predict_cxr(self, img_path):
        x = self._preprocess_cxr(img_path).to(self.device)
        with torch.no_grad():
            recon = self.cxr_model(x)
            mse = torch.mean((recon - x) ** 2, dim=(1, 2, 3)).item()
        is_abnormal = mse > self.cxr_thresh
        return {
            "modality": "CXR",
            "mse": mse,
            "threshold": self.cxr_thresh,
            "is_abnormal": is_abnormal,
            "recommendation": "⚠️ 检测到异常，建议就医" if is_abnormal else "✅ 胸片正常"
        }


def main():
    import argparse
    parser = argparse.ArgumentParser(description="先天性心脏病多模态异常检测推理")
    parser.add_argument("--ecg", help="ECG文件路径 (不含扩展名)")
    parser.add_argument("--pcg", help="PCG WAV文件路径")
    parser.add_argument("--cxr", help="CXR图像文件路径")
    args = parser.parse_args()

    detector = CHDDetector()

    if args.ecg:
        result = detector.predict_ecg(args.ecg)
        print(f"\n{result['modality']} 检测结果:")
        print(f"  重构误差 (MSE): {result['mse']:.6f}")
        print(f"  正常阈值: {result['threshold']:.6f}")
        print(f"  {result['recommendation']}")

    if args.pcg:
        result = detector.predict_pcg(args.pcg)
        print(f"\n{result['modality']} 检测结果:")
        print(f"  重构误差 (MSE): {result['mse']:.6f}")
        print(f"  正常阈值: {result['threshold']:.6f}")
        print(f"  {result['recommendation']}")

    if args.cxr:
        result = detector.predict_cxr(args.cxr)
        print(f"\n{result['modality']} 检测结果:")
        print(f"  重构误差 (MSE): {result['mse']:.6f}")
        print(f"  正常阈值: {result['threshold']:.6f}")
        print(f"  {result['recommendation']}")


if __name__ == "__main__":
    main()
