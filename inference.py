import argparse
from pathlib import Path

import librosa
import numpy as np
import torch
import wfdb
from PIL import Image
from torchvision import transforms

import config
from model import ECGAutoencoder, PCGAutoencoder, CXRAutoencoder


DISCLAIMER = (
    "This system is for research, teaching, and demonstration only. "
    "It must not replace physician diagnosis or medical decision-making."
)
FUSION_WEIGHTS = {
    "ECG": 0.50,
    "PCG": 0.30,
    "CXR": 0.20,
}
EXPECTED_MODALITIES = list(FUSION_WEIGHTS.keys())


def load_state_dict_compat(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def risk_level(score):
    if score is None:
        return "unknown"
    if score >= 1.50:
        return "high"
    if score >= 1.00:
        return "elevated"
    return "low"


def fuse_results(results):
    if not results:
        return {
            "available_modalities": [],
            "missing_modalities": EXPECTED_MODALITIES,
            "fusion_score": None,
            "risk_level": "unknown",
            "is_abnormal": None,
        }

    by_modality = {item["modality"]: item for item in results}
    available = [name for name in EXPECTED_MODALITIES if name in by_modality]
    missing = [name for name in EXPECTED_MODALITIES if name not in by_modality]
    weight_sum = sum(FUSION_WEIGHTS[name] for name in available)
    normalized_weights = {
        name: FUSION_WEIGHTS[name] / weight_sum for name in available
    }
    fusion_score = sum(
        by_modality[name]["score"] * normalized_weights[name] for name in available
    )
    return {
        "method": "weighted_late_fusion",
        "available_modalities": available,
        "missing_modalities": missing,
        "weights": normalized_weights,
        "raw_weights": FUSION_WEIGHTS,
        "fusion_score": fusion_score,
        "max_single_modality_score": max(by_modality[name]["score"] for name in available),
        "abnormal_modalities": [name for name in available if by_modality[name]["is_abnormal"]],
        "risk_level": risk_level(fusion_score),
        "is_abnormal": fusion_score >= 1.0,
    }


class CHDDetector:
    def __init__(self, model_dir=None):
        self.model_dir = Path(model_dir or Path(__file__).resolve().parent)
        self.device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")
        self._load_models()
        self._load_thresholds()

    def _load_models(self):
        self.ecg_model = ECGAutoencoder().to(self.device)
        self.ecg_model.load_state_dict(
            load_state_dict_compat(self.model_dir / "ecg_ae.pth", self.device)
        )
        self.ecg_model.eval()

        self.pcg_model = PCGAutoencoder().to(self.device)
        self.pcg_model.load_state_dict(
            load_state_dict_compat(self.model_dir / "pcg_ae.pth", self.device)
        )
        self.pcg_model.eval()

        self.cxr_model = CXRAutoencoder().to(self.device)
        self.cxr_model.load_state_dict(
            load_state_dict_compat(self.model_dir / "cxr_ae.pth", self.device)
        )
        self.cxr_model.eval()

    def _load_thresholds(self):
        thresh_data = np.load(self.model_dir / "thresholds.npz")
        self.ecg_thresh = float(thresh_data["ecg"])
        self.pcg_thresh = float(thresh_data["pcg"])
        self.cxr_thresh = float(thresh_data["cxr"])

    def _preprocess_ecg(self, ecg_path):
        sig, _ = wfdb.rdsamp(str(ecg_path))
        if sig.shape[0] < 1000:
            pad_len = 1000 - sig.shape[0]
            sig = np.pad(sig, ((0, pad_len), (0, 0)), mode="constant")
        sig = sig[:1000, :12].T
        sig = (sig - sig.mean(axis=1, keepdims=True)) / (
            sig.std(axis=1, keepdims=True) + 1e-8
        )
        return torch.tensor(sig, dtype=torch.float32).unsqueeze(0)

    def _preprocess_pcg(self, wav_path):
        y, sr = librosa.load(wav_path, sr=2000)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, n_fft=512, hop_length=256)
        if mfcc.shape[1] < 200:
            mfcc = np.pad(mfcc, ((0, 0), (0, 200 - mfcc.shape[1])), mode="constant")
        else:
            mfcc = mfcc[:, :200]
        mfcc = (mfcc - mfcc.mean()) / (mfcc.std() + 1e-8)
        return torch.tensor(mfcc, dtype=torch.float32).unsqueeze(0)

    def _preprocess_cxr(self, img_path):
        transform = transforms.Compose(
            [
                transforms.Resize((256, 256)),
                transforms.Grayscale(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5]),
            ]
        )
        img = Image.open(img_path).convert("L")
        return transform(img).unsqueeze(0)

    def predict_ecg(self, ecg_path):
        x = self._preprocess_ecg(ecg_path).to(self.device)
        with torch.no_grad():
            recon = self.ecg_model(x)
            mse = torch.mean((recon - x) ** 2, dim=(1, 2)).item()
        return self._format_result("ECG", mse, self.ecg_thresh)

    def predict_pcg(self, wav_path):
        x = self._preprocess_pcg(wav_path).to(self.device)
        with torch.no_grad():
            recon = self.pcg_model(x)
            mse = torch.mean((recon - x) ** 2, dim=(1, 2)).item()
        return self._format_result("PCG", mse, self.pcg_thresh)

    def predict_cxr(self, img_path):
        x = self._preprocess_cxr(img_path).to(self.device)
        with torch.no_grad():
            recon = self.cxr_model(x)
            mse = torch.mean((recon - x) ** 2, dim=(1, 2, 3)).item()
        return self._format_result("CXR", mse, self.cxr_thresh)

    def _format_result(self, modality, mse, threshold):
        score = mse / threshold if threshold > 0 else float("inf")
        return {
            "modality": modality,
            "mse": mse,
            "threshold": threshold,
            "score": score,
            "risk_level": risk_level(score),
            "is_abnormal": mse > threshold,
        }


def print_result(result):
    status = "abnormal" if result["is_abnormal"] else "below threshold"
    print(f"\n{result['modality']} result")
    print(f"  status: {status}")
    print(f"  mse: {result['mse']:.6f}")
    print(f"  threshold: {result['threshold']:.6f}")
    print(f"  normalized score: {result['score']:.3f}")
    print(f"  risk level: {result['risk_level']}")


def print_fusion(fusion):
    print("\nWeighted late fusion")
    print(f"  available modalities: {', '.join(fusion['available_modalities'])}")
    if fusion["missing_modalities"]:
        print(f"  missing modalities: {', '.join(fusion['missing_modalities'])}")
    print(f"  normalized weights: {fusion['weights']}")
    print(f"  fusion score: {fusion['fusion_score']:.3f}")
    print(f"  risk level: {fusion['risk_level']}")
    print(f"  abnormal: {fusion['is_abnormal']}")
    if fusion["abnormal_modalities"]:
        print(f"  abnormal modalities: {', '.join(fusion['abnormal_modalities'])}")
    print(f"  disclaimer: {DISCLAIMER}")


def main():
    parser = argparse.ArgumentParser(description="CHD multimodal anomaly detection inference")
    parser.add_argument("--ecg", help="WFDB ECG record path without .hea/.dat extension")
    parser.add_argument("--pcg", help="PCG WAV file path")
    parser.add_argument("--cxr", help="CXR image file path")
    parser.add_argument("--model-dir", default=None, help="Directory containing model weights and thresholds.npz")
    args = parser.parse_args()

    detector = CHDDetector(model_dir=args.model_dir)
    results = []

    if args.ecg:
        results.append(detector.predict_ecg(args.ecg))

    if args.pcg:
        results.append(detector.predict_pcg(args.pcg))

    if args.cxr:
        results.append(detector.predict_cxr(args.cxr))

    if not results:
        parser.error("Provide at least one of --ecg, --pcg, or --cxr.")

    for result in results:
        print_result(result)

    print_fusion(fuse_results(results))


if __name__ == "__main__":
    main()
