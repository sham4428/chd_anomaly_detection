import io
import os
import tempfile
from pathlib import Path

import librosa
import numpy as np
import torch
import wfdb
from flask import Flask, jsonify, request
from flask_cors import CORS
from PIL import Image
from torchvision import transforms

from model import ECGAutoencoder, PCGAutoencoder, CXRAutoencoder


app = Flask(__name__)
CORS(app)


DISCLAIMER = (
    "This system is for research, teaching, and demonstration only. "
    "It must not replace physician diagnosis or medical decision-making."
)

# Late-fusion weights are based on the current evaluation summary:
# ECG has the strongest ROC AUC, PCG is weaker, and CXR is currently low quality.
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


def recommendation(is_abnormal, modality_name):
    if is_abnormal:
        return (
            f"{modality_name} reconstruction error is above its threshold. "
            "Further clinical assessment is recommended."
        )
    return f"{modality_name} reconstruction error is not above its threshold."


def fuse_results(results):
    if not results:
        return {
            "method": "weighted_late_fusion",
            "available_modalities": [],
            "missing_modalities": EXPECTED_MODALITIES,
            "abnormal_modalities": [],
            "weights": {},
            "raw_weights": FUSION_WEIGHTS,
            "fusion_score": None,
            "max_single_modality_score": None,
            "risk_level": "unknown",
            "is_abnormal": None,
            "recommendation": "No analyzable modality data was provided.",
            "disclaimer": DISCLAIMER,
        }

    by_modality = {item["modality"]: item for item in results}
    available = [name for name in EXPECTED_MODALITIES if name in by_modality]
    missing = [name for name in EXPECTED_MODALITIES if name not in by_modality]
    available_weight_sum = sum(FUSION_WEIGHTS[name] for name in available)
    normalized_weights = {
        name: FUSION_WEIGHTS[name] / available_weight_sum for name in available
    }

    fusion_score = sum(
        by_modality[name]["score"] * normalized_weights[name] for name in available
    )
    max_single_score = max(by_modality[name]["score"] for name in available)
    abnormal_modalities = [
        name for name in available if by_modality[name]["is_abnormal"]
    ]
    level = risk_level(fusion_score)
    is_abnormal = fusion_score >= 1.0

    if is_abnormal:
        text = (
            "Weighted late fusion indicates elevated CHD screening risk. "
            "Please combine this result with clinical examination."
        )
    elif abnormal_modalities:
        text = (
            "At least one modality is abnormal, but the weighted fusion score is below "
            "the abnormal threshold. Review the abnormal modality and clinical context."
        )
    else:
        text = (
            "The available modalities are below the weighted fusion threshold. "
            "Clinical symptoms or known risk factors should still take priority."
        )

    return {
        "method": "weighted_late_fusion",
        "available_modalities": available,
        "missing_modalities": missing,
        "abnormal_modalities": abnormal_modalities,
        "weights": normalized_weights,
        "raw_weights": FUSION_WEIGHTS,
        "fusion_score": fusion_score,
        "max_single_modality_score": max_single_score,
        "risk_level": level,
        "is_abnormal": is_abnormal,
        "recommendation": text,
        "disclaimer": DISCLAIMER,
    }


class CHDDetector:
    def __init__(self, model_dir=None):
        self.model_dir = Path(model_dir or Path(__file__).resolve().parent)
        self.device = torch.device("cpu")
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

    def preprocess_ecg_record(self, record_path):
        sig, _ = wfdb.rdsamp(str(record_path))
        if sig.shape[0] < 1000:
            pad_len = 1000 - sig.shape[0]
            sig = np.pad(sig, ((0, pad_len), (0, 0)), mode="constant")
        sig = sig[:1000, :12].T
        sig = (sig - sig.mean(axis=1, keepdims=True)) / (
            sig.std(axis=1, keepdims=True) + 1e-8
        )
        return sig

    def preprocess_pcg_file(self, file_obj):
        y, sr = librosa.load(file_obj, sr=2000)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, n_fft=512, hop_length=256)
        if mfcc.shape[1] < 200:
            mfcc = np.pad(mfcc, ((0, 0), (0, 200 - mfcc.shape[1])), mode="constant")
        else:
            mfcc = mfcc[:, :200]
        return (mfcc - mfcc.mean()) / (mfcc.std() + 1e-8)

    def preprocess_cxr_file(self, file_obj):
        img = Image.open(file_obj).convert("L")
        transform = transforms.Compose(
            [
                transforms.Resize((256, 256)),
                transforms.Grayscale(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5]),
            ]
        )
        return transform(img)

    def predict_ecg_array(self, ecg_data):
        x = torch.tensor(ecg_data, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            recon = self.ecg_model(x)
            mse = torch.mean((recon - x) ** 2, dim=(1, 2)).item()
        return self._format_result("ECG", mse, self.ecg_thresh)

    def predict_pcg_array(self, mfcc_data):
        x = torch.tensor(mfcc_data, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            recon = self.pcg_model(x)
            mse = torch.mean((recon - x) ** 2, dim=(1, 2)).item()
        return self._format_result("PCG", mse, self.pcg_thresh)

    def predict_cxr_tensor(self, img_tensor):
        x = img_tensor.unsqueeze(0).to(self.device)
        with torch.no_grad():
            recon = self.cxr_model(x)
            mse = torch.mean((recon - x) ** 2, dim=(1, 2, 3)).item()
        return self._format_result("CXR", mse, self.cxr_thresh)

    def _format_result(self, modality, mse, threshold):
        score = mse / threshold if threshold > 0 else float("inf")
        is_abnormal = mse > threshold
        return {
            "modality": modality,
            "mse": mse,
            "threshold": threshold,
            "score": score,
            "risk_level": risk_level(score),
            "is_abnormal": is_abnormal,
            "recommendation": recommendation(is_abnormal, modality),
            "disclaimer": DISCLAIMER,
        }


detector = CHDDetector()


def error_response(message, status=400):
    return jsonify({"error": message, "disclaimer": DISCLAIMER}), status


def ecg_from_request(hea_key="hea", dat_key="dat", record_path_key="record_path"):
    payload = request.get_json(silent=True) or {}
    record_path = request.form.get(record_path_key) or payload.get(record_path_key)
    if record_path:
        return detector.preprocess_ecg_record(record_path)

    hea_file = request.files.get(hea_key)
    dat_file = request.files.get(dat_key)
    if not hea_file or not dat_file:
        raise ValueError("ECG requires .hea and .dat uploads, or a record_path value.")

    with tempfile.TemporaryDirectory() as tmpdir:
        hea_name = Path(hea_file.filename or "record.hea").name
        stem = Path(hea_name).stem or "record"
        hea_path = Path(tmpdir) / f"{stem}.hea"
        dat_path = Path(tmpdir) / f"{stem}.dat"
        hea_file.save(hea_path)
        dat_file.save(dat_path)
        return detector.preprocess_ecg_record(hea_path.with_suffix(""))


@app.route("/")
def index():
    return jsonify(
        {
            "service": "CHD multimodal anomaly detection API",
            "version": "1.2",
            "modalities": EXPECTED_MODALITIES,
            "fusion": {
                "method": "weighted_late_fusion",
                "weights": FUSION_WEIGHTS,
                "missing_modality_policy": "renormalize_available_weights",
                "threshold": 1.0,
            },
            "endpoints": {
                "/api/health": "health check",
                "/api/fusion/config": "fusion configuration",
                "/api/predict/ecg": "ECG prediction",
                "/api/predict/pcg": "PCG prediction",
                "/api/predict/cxr": "CXR prediction",
                "/api/predict/multimodal": "weighted late-fusion prediction",
            },
            "disclaimer": DISCLAIMER,
        }
    )


@app.route("/api/health", methods=["GET"])
def health_check():
    return jsonify({"status": "ok", "message": "service is running", "disclaimer": DISCLAIMER})


@app.route("/api/fusion/config", methods=["GET"])
def fusion_config():
    return jsonify(
        {
            "method": "weighted_late_fusion",
            "weights": FUSION_WEIGHTS,
            "score_definition": "modality_mse / modality_threshold",
            "fusion_score": "sum(normalized_weight * modality_score)",
            "abnormal_threshold": 1.0,
            "risk_levels": {
                "low": "score < 1.0",
                "elevated": "1.0 <= score < 1.5",
                "high": "score >= 1.5",
            },
            "missing_modality_policy": "Weights are renormalized over available modalities.",
            "rationale": "ECG receives the highest weight because it has the strongest current ROC AUC; CXR is down-weighted because current CXR performance is weak.",
            "disclaimer": DISCLAIMER,
        }
    )


@app.route("/api/predict/ecg", methods=["POST"])
def predict_ecg():
    try:
        ecg_data = ecg_from_request()
        return jsonify(detector.predict_ecg_array(ecg_data))
    except Exception as exc:
        return error_response(str(exc), 500)


@app.route("/api/predict/pcg", methods=["POST"])
def predict_pcg():
    try:
        file = request.files.get("file")
        if not file:
            return error_response("No PCG file uploaded.")

        mfcc = detector.preprocess_pcg_file(io.BytesIO(file.read()))
        return jsonify(detector.predict_pcg_array(mfcc))
    except Exception as exc:
        return error_response(str(exc), 500)


@app.route("/api/predict/cxr", methods=["POST"])
def predict_cxr():
    try:
        file = request.files.get("file")
        if not file:
            return error_response("No CXR image uploaded.")

        img_tensor = detector.preprocess_cxr_file(io.BytesIO(file.read()))
        return jsonify(detector.predict_cxr_tensor(img_tensor))
    except Exception as exc:
        return error_response(str(exc), 500)


@app.route("/api/predict/multimodal", methods=["POST"])
def predict_multimodal():
    results = []

    try:
        payload = request.get_json(silent=True) or {}
        if request.files.get("ecg_hea") and request.files.get("ecg_dat"):
            ecg_data = ecg_from_request("ecg_hea", "ecg_dat", "ecg_record_path")
            results.append(detector.predict_ecg_array(ecg_data))
        elif request.form.get("ecg_record_path") or payload.get("ecg_record_path"):
            ecg_data = ecg_from_request("ecg_hea", "ecg_dat", "ecg_record_path")
            results.append(detector.predict_ecg_array(ecg_data))

        pcg_file = request.files.get("pcg_file")
        if pcg_file:
            mfcc = detector.preprocess_pcg_file(io.BytesIO(pcg_file.read()))
            results.append(detector.predict_pcg_array(mfcc))

        cxr_file = request.files.get("cxr_file")
        if cxr_file:
            img_tensor = detector.preprocess_cxr_file(io.BytesIO(cxr_file.read()))
            results.append(detector.predict_cxr_tensor(img_tensor))

        if not results:
            return error_response("No analyzable ECG, PCG, or CXR data was provided.")

        return jsonify({"fusion": fuse_results(results), "results": results})
    except Exception as exc:
        return error_response(str(exc), 500)


if __name__ == "__main__":
    port = int(os.getenv("PORT", "7860"))
    app.run(host="0.0.0.0", port=port, debug=False)
