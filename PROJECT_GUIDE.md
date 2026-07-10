# CHD Anomaly Detection Project Guide

This file is a practical map of the project. Read this before changing code.

## 1. What This Project Does

The project is a multimodal CHD screening demo.

It handles three modalities:

- ECG: 12 leads x 1000 points
- PCG: 13 MFCC features x 200 frames
- CXR: grayscale chest X-ray resized to 256 x 256

Each modality has its own autoencoder. The autoencoder is trained mostly on normal samples. At inference time:

1. Input is reconstructed by the model.
2. Reconstruction error is computed.
3. If error is above the modality threshold, that modality is marked abnormal.
4. API late-fuses available modality scores.

This is anomaly detection, not a validated clinical diagnostic model.

## 2. Important Files

Core files:

- `config.py`: paths and training hyperparameters.
- `data_preprocessing.py`: creates `.npy` data files under `data/`.
- `model.py`: ECG, PCG, and CXR autoencoder definitions.
- `train.py`: trains the three autoencoders on normal data.
- `optimize_thresholds.py`: selects thresholds on a calibration split and reports held-out test metrics.
- `evaluate_model.py`: evaluates individual modality models and plots error/ROC results.
- `app.py`: Flask API for ECG, PCG, CXR, and multimodal prediction.

Generated artifacts:

- `ecg_ae.pth`, `pcg_ae.pth`, `cxr_ae.pth`: trained model weights.
- `thresholds.npz`: modality thresholds.
- `evaluation_results/`: evaluation reports and plots.

## 3. Data Files

Preprocessed data lives in `data/`.

Training-normal files:

- `normal_ecg.npy`: normal ECG samples.
- `normal_pcg.npy`: normal PCG samples.
- `normal_cxr.npy`: normal CXR samples.

Labeled evaluation files:

- `ecg_data.npy`, `ecg_labels.npy`
- `pcg_data.npy`, `pcg_labels.npy`
- `cxr_data.npy`, `cxr_labels.npy`

Current local sample counts:

- ECG normal: 5000
- PCG normal: 500
- CXR normal: 208

`train.py` now trains each modality separately, so ECG and PCG are no longer truncated to the CXR sample count.

## 4. Current Model State

`model.py` has been changed recently:

- ECG now has a latent bottleneck.
- PCG now has a latent bottleneck.
- CXR now uses a real CNN decoder after the Transformer encoder.

Because model structure changed, old `.pth` files may not load into the new code.

You should retrain before evaluating or serving:

```powershell
cd D:\chd_anomaly_detection
python train.py --epochs 50
python optimize_thresholds.py
python evaluate_model.py
```

If training is too slow, test with fewer epochs first:

```powershell
python train.py --epochs 1
```

## 5. Correct Evaluation Order

Use this order:

1. Preprocess data if needed.
2. Train models on normal training data.
3. Optimize thresholds on calibration data only.
4. Report metrics on held-out test data.
5. Start API only after compatible weights and thresholds exist.

Do not use the same labeled test set both to tune thresholds and report final metrics.

## 6. Threshold Logic

`optimize_thresholds.py` now does this:

- Split labeled data into calibration and held-out test.
- Choose threshold only on calibration.
- Report held-out test metrics without changing the threshold.

Reported metrics include:

- sensitivity / recall
- specificity
- PPV / precision
- NPV
- ROC-AUC
- PR-AUC
- FPR at 90% sensitivity
- MSE AUC and inverse-MSE AUC direction check

If inverse-MSE AUC is better than MSE AUC, the anomaly score direction may be wrong.

## 7. API Notes

`app.py` is now lazy-loaded:

- Importing the API file should not immediately load model weights.
- Models load on first prediction request.
- Missing model files return a structured API error.

Fusion output includes:

- `fusion_score`: weighted normalized score.
- `confidence`: sum of raw weights for available modalities.
- `available_modalities`
- `missing_modalities`

If only one weak modality is available, confidence should be low.

## 8. What To Ignore For Now

These files are secondary or old experiment scripts:

- `simple_eval.py`
- `evaluate_parallel.py`
- `full_test.py`
- `analyze_results.py`
- `update_thresholds.py`
- `update_models.py`

They may still run, but the main path should be:

```text
data_preprocessing.py -> train.py -> optimize_thresholds.py -> evaluate_model.py -> app.py
```

## 9. Known Limitations

- CXR autoencoder is still a weak approach for medical imaging anomaly detection.
- Fusion weights are still fixed, not learned from calibration data.
- Patient-level split is not implemented yet.
- Bootstrap confidence intervals are not implemented yet.
- External validation is not implemented.

The next sensible improvements are:

1. Add patient IDs to preprocessing outputs.
2. Implement patient-level train/calibration/test split.
3. Add `evaluate_fusion.py`.
4. Learn fusion weights on calibration data.
5. Replace CXR autoencoder with pretrained feature extraction plus anomaly detection or supervised classification.
