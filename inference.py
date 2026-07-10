import argparse
from pathlib import Path

from app import CHDDetector, DISCLAIMER, fuse_results


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
    print(f"  input coverage confidence: {fusion['confidence']:.2f}")
    print(f"  fusion score: {fusion['fusion_score']:.3f}")
    print(f"  risk level: {fusion['risk_level']}")
    print(f"  abnormal: {fusion['is_abnormal']}")
    if fusion["abnormal_modalities"]:
        print(f"  abnormal modalities: {', '.join(fusion['abnormal_modalities'])}")
    print(f"  disclaimer: {DISCLAIMER}")


def main():
    parser = argparse.ArgumentParser(
        description="CHD multimodal anomaly detection inference"
    )
    parser.add_argument("--ecg", help="WFDB ECG record path without .hea/.dat extension")
    parser.add_argument("--pcg", help="PCG audio file path")
    parser.add_argument("--cxr", help="CXR image file path")
    parser.add_argument(
        "--model-dir",
        default=None,
        help="Directory containing model weights and thresholds.npz",
    )
    args = parser.parse_args()

    detector = CHDDetector(model_dir=args.model_dir)
    print(f"Loaded checkpoints: {detector.model_architectures}")
    results = []

    if args.ecg:
        ecg_data = detector.preprocess_ecg_record(args.ecg)
        results.append(detector.predict_ecg_array(ecg_data))

    if args.pcg:
        with Path(args.pcg).open("rb") as file:
            pcg_data = detector.preprocess_pcg_file(file)
        results.append(detector.predict_pcg_array(pcg_data))

    if args.cxr:
        with Path(args.cxr).open("rb") as file:
            cxr_data = detector.preprocess_cxr_file(file)
        results.append(detector.predict_cxr_tensor(cxr_data))

    if not results:
        parser.error("Provide at least one of --ecg, --pcg, or --cxr.")

    for result in results:
        print_result(result)
    print_fusion(fuse_results(results))


if __name__ == "__main__":
    main()
