import json

import numpy as np
import torch

from model import CXRAutoencoder, ECGAutoencoder, PCGAutoencoder, load_autoencoder


def export_model(model_cls, model_path, output_path, input_shape, modality):
    device = torch.device("cpu")
    model = load_autoencoder(model_cls, model_path, device)
    dummy_input = torch.randn(*input_shape, device=device)

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"},
        },
    )
    print(
        f"[OK] {modality} model exported to {output_path} "
        f"({model.checkpoint_architecture})"
    )


def export_ecg_onnx(model_path="ecg_ae.pth", output_path="ecg_ae.onnx"):
    export_model(ECGAutoencoder, model_path, output_path, (1, 12, 1000), "ECG")


def export_pcg_onnx(model_path="pcg_ae.pth", output_path="pcg_ae.onnx"):
    export_model(PCGAutoencoder, model_path, output_path, (1, 13, 200), "PCG")


def export_cxr_onnx(model_path="cxr_ae.pth", output_path="cxr_ae.onnx"):
    export_model(CXRAutoencoder, model_path, output_path, (1, 1, 256, 256), "CXR")


def export_thresholds(thresholds_path="thresholds.npz", output_path="thresholds.json"):
    data = np.load(thresholds_path)
    thresholds = {
        "ecg": float(data["ecg"]),
        "pcg": float(data["pcg"]),
        "cxr": float(data["cxr"]),
    }
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(thresholds, file, indent=2)
    print(f"[OK] Thresholds exported to {output_path}")


if __name__ == "__main__":
    print("[START] Exporting ONNX models")
    export_ecg_onnx()
    export_pcg_onnx()
    export_cxr_onnx()
    export_thresholds()
    print("[DONE] ONNX export completed")
