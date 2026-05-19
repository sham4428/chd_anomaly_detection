import os
import ast
import numpy as np
import pandas as pd
import librosa
import wfdb
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
import config


def preprocess_ecg(ptbxl_path, output_dir, n_samples=5000):
    """预处理PTB-XL心电数据，使用scp_codes筛选正常人"""
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(ptbxl_path, "ptbxl_database.csv")
    df = pd.read_csv(csv_path)

    print(f"[INFO] PTB-XL 总记录数: {len(df)}")

    def get_diagnostic_class(scp_str):
        try:
            scp_dict = ast.literal_eval(scp_str)
            if 'NORM' in scp_dict and scp_dict['NORM'] >= 80.0:
                return 'NORM'
            return None
        except:
            return None

    df['diagnostic_class'] = df['scp_codes'].apply(get_diagnostic_class)
    normal_df = df[df['diagnostic_class'] == 'NORM']
    print(f"[OK] 筛选到正常 ECG 记录: {len(normal_df)} 条")

    if len(normal_df) == 0:
        print("[ERROR] 未找到正常 ECG，请检查 scp_codes 列。")
        return

    if len(normal_df) > n_samples:
        normal_df = normal_df.sample(n_samples, random_state=42)

    ecg_list = []
    for idx, row in tqdm(normal_df.iterrows(), total=len(normal_df), desc="ECG"):
        try:
            file_path = os.path.join(ptbxl_path, row["filename_lr"])
            sig, _ = wfdb.rdsamp(file_path)
            sig = sig[:1000, :12].T
            sig = (sig - sig.mean(axis=1, keepdims=True)) / (sig.std(axis=1, keepdims=True) + 1e-8)
            ecg_list.append(sig)
        except Exception as e:
            continue

    np.save(os.path.join(output_dir, "normal_ecg.npy"), np.array(ecg_list))
    print(f"[SAVE] ECG 保存 {len(ecg_list)} 例")


def preprocess_pcg(circor_path, output_dir, n_samples=500):
    """预处理 CirCor 心音数据，筛选无杂音样本"""
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(circor_path, "training_data.csv")
    df = pd.read_csv(csv_path)

    print(f"[INFO] CirCor 总记录数: {len(df)}")
    normal_df = df[df['Murmur'] == 'Absent']
    print(f"[OK] 筛选到正常 PCG: {len(normal_df)} 人")

    data_dir = os.path.join(circor_path, "training_data")
    if not os.path.exists(data_dir):
        data_dir = circor_path

    pcg_list = []
    for _, row in tqdm(normal_df.iterrows(), total=len(normal_df), desc="PCG"):
        patient_id = str(row['Patient ID'])
        wav_files = [f for f in os.listdir(data_dir) if f.startswith(patient_id) and f.endswith('.wav')]
        for wav in wav_files[:2]:
            if len(pcg_list) >= n_samples:
                break
            try:
                y, sr = librosa.load(os.path.join(data_dir, wav), sr=2000)
                mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, n_fft=512, hop_length=256)
                if mfcc.shape[1] < 200:
                    mfcc = np.pad(mfcc, ((0, 0), (0, 200 - mfcc.shape[1])), mode='constant')
                else:
                    mfcc = mfcc[:, :200]
                mfcc = (mfcc - mfcc.mean()) / (mfcc.std() + 1e-8)
                pcg_list.append(mfcc)
            except Exception as e:
                continue
        if len(pcg_list) >= n_samples:
            break

    np.save(os.path.join(output_dir, "normal_pcg.npy"), np.array(pcg_list))
    print(f"[SAVE] PCG 保存 {len(pcg_list)} 例")


def preprocess_cxr(chd_cxr_path, output_dir, n_samples=5000):
    """预处理CHD胸片数据，仅提取Normal类别"""
    os.makedirs(output_dir, exist_ok=True)
    normal_dir = os.path.join(chd_cxr_path, "Normal")

    if not os.path.exists(normal_dir):
        print(f"[ERROR] 未找到 Normal 目录")
        return

    image_files = [f for f in os.listdir(normal_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if len(image_files) > n_samples:
        image_files = image_files[:n_samples]
    print(f"[INFO] 找到正常胸片图像: {len(image_files)} 张")

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    cxr_list = []
    for fname in tqdm(image_files, desc="CXR"):
        try:
            img = Image.open(os.path.join(normal_dir, fname)).convert("L")
            img_tensor = transform(img).squeeze(0).numpy()
            cxr_list.append(img_tensor)
        except Exception as e:
            continue

    np.save(os.path.join(output_dir, "normal_cxr.npy"), np.array(cxr_list))
    print(f"[SAVE] CXR 保存 {len(cxr_list)} 例")


if __name__ == "__main__":
    print("[START] 开始数据预处理...")
    preprocess_ecg(config.PTBXL_PATH, config.OUTPUT_DIR)
    preprocess_pcg(config.CIRCOR_PATH, config.OUTPUT_DIR)
    preprocess_cxr(config.CHD_CXR_PATH, config.OUTPUT_DIR)
    print("[DONE] 所有预处理完成！")
