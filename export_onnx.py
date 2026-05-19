import os
import torch
import numpy as np
from model import ECGAutoencoder, PCGAutoencoder, CXRAutoencoder
import config


def export_ecg_onnx(model_path="ecg_ae.pth", output_path="ecg_ae.onnx"):
    device = torch.device('cpu')
    model = ECGAutoencoder().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    dummy_input = torch.randn(1, 12, 1000).to(device)

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    print(f"[OK] ECG Autoencoder 导出到 {output_path}")


def export_pcg_onnx(model_path="pcg_ae.pth", output_path="pcg_ae.onnx"):
    device = torch.device('cpu')
    model = PCGAutoencoder().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    dummy_input = torch.randn(1, 13, 200).to(device)

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    print(f"[OK] PCG Autoencoder 导出到 {output_path}")


def export_cxr_onnx(model_path="cxr_ae.pth", output_path="cxr_ae.onnx"):
    device = torch.device('cpu')
    model = CXRAutoencoder().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    dummy_input = torch.randn(1, 1, 256, 256).to(device)

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    print(f"[OK] CXR Autoencoder 导出到 {output_path}")


def export_thresholds(thresholds_path="thresholds.npz", output_path="thresholds.json"):
    import json
    data = np.load(thresholds_path)
    thresholds = {
        "ecg": float(data["ecg"]),
        "pcg": float(data["pcg"]),
        "cxr": float(data["cxr"])
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(thresholds, f, indent=2, ensure_ascii=False)
    print(f"[OK] 阈值导出到 {output_path}")


if __name__ == "__main__":
    print("[START] 开始导出ONNX模型")

    export_ecg_onnx()
    export_pcg_onnx()
    export_cxr_onnx()
    export_thresholds()

    print("\n[DONE] 所有模型导出完成！")
    print("生成的文件:")
    print("  - ecg_ae.onnx")
    print("  - pcg_ae.onnx")
    print("  - cxr_ae.onnx")
    print("  - thresholds.json")
