import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from model import ECGAutoencoder, PCGAutoencoder, CXRAutoencoder
from tqdm import tqdm
import config


class MultiModalDataset(Dataset):
    def __init__(self, ecg_path, pcg_path, cxr_path):
        self.ecg = np.load(ecg_path)
        self.pcg = np.load(pcg_path)
        self.cxr = np.load(cxr_path)
        self.min_len = min(len(self.ecg), len(self.pcg), len(self.cxr))
        print(f"统一使用 {self.min_len} 例数据")

    def __len__(self):
        return self.min_len

    def __getitem__(self, idx):
        return {
            "ecg": torch.tensor(self.ecg[idx], dtype=torch.float32),
            "pcg": torch.tensor(self.pcg[idx], dtype=torch.float32),
            "cxr": torch.tensor(self.cxr[idx], dtype=torch.float32).unsqueeze(0)
        }


def train_one_model(model, loader, criterion, optimizer, device, epochs, name):
    model.train()
    best_loss = float('inf')
    for epoch in range(epochs):
        total_loss = 0
        for batch in tqdm(loader, desc=f"{name} Epoch {epoch + 1}"):
            x = batch[name].to(device)
            optimizer.zero_grad()
            recon = model(x)
            loss = criterion(recon, x)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(loader)
        print(f"[{name}] Epoch {epoch + 1}: Loss={avg_loss:.6f}")
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), f"{name}_best.pth")
    torch.save(model.state_dict(), f"{name}_final.pth")
    return model


def compute_threshold(model, loader, device, name):
    model.eval()
    errors = []
    with torch.no_grad():
        for batch in loader:
            x = batch[name].to(device)
            recon = model(x)
            if name == "cxr":
                err = torch.mean((recon - x) ** 2, dim=(1, 2, 3))
            else:
                err = torch.mean((recon - x) ** 2, dim=(1, 2))
            errors.extend(err.cpu().numpy())
    thresh = np.percentile(errors, 95)
    print(f"[{name}] 阈值 (95%): {thresh:.6f}")
    return thresh


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=config.EPOCHS)
    parser.add_argument("--batch_size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=config.LEARNING_RATE)
    args = parser.parse_args()

    device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    dataset = MultiModalDataset(
        os.path.join(config.OUTPUT_DIR, "normal_ecg.npy"),
        os.path.join(config.OUTPUT_DIR, "normal_pcg.npy"),
        os.path.join(config.OUTPUT_DIR, "normal_cxr.npy")
    )
    train_data, val_data = train_test_split(dataset, test_size=0.2, random_state=42)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False, num_workers=0)

    criterion = nn.MSELoss()

    print("\n训练 ECG Autoencoder")
    ecg_model = ECGAutoencoder().to(device)
    ecg_model = train_one_model(ecg_model, train_loader, criterion,
                                optim.Adam(ecg_model.parameters(), lr=args.lr),
                                device, args.epochs, "ecg")

    print("\n训练 PCG Autoencoder")
    pcg_model = PCGAutoencoder().to(device)
    pcg_model = train_one_model(pcg_model, train_loader, criterion,
                                optim.Adam(pcg_model.parameters(), lr=args.lr),
                                device, args.epochs, "pcg")

    print("\n训练 CXR Autoencoder")
    cxr_model = CXRAutoencoder().to(device)
    cxr_model = train_one_model(cxr_model, train_loader, criterion,
                                optim.Adam(cxr_model.parameters(), lr=args.lr),
                                device, args.epochs, "cxr")

    ecg_model.load_state_dict(torch.load("ecg_best.pth", map_location=device, weights_only=True))
    pcg_model.load_state_dict(torch.load("pcg_best.pth", map_location=device, weights_only=True))
    cxr_model.load_state_dict(torch.load("cxr_best.pth", map_location=device, weights_only=True))

    ecg_thresh = compute_threshold(ecg_model, val_loader, device, "ecg")
    pcg_thresh = compute_threshold(pcg_model, val_loader, device, "pcg")
    cxr_thresh = compute_threshold(cxr_model, val_loader, device, "cxr")
    
    np.savez("thresholds.npz", ecg=ecg_thresh, pcg=pcg_thresh, cxr=cxr_thresh)
    
    # 重命名模型文件，如果目标已存在则删除
    for old_name, new_name in [("ecg_best.pth", "ecg_ae.pth"), 
                               ("pcg_best.pth", "pcg_ae.pth"), 
                               ("cxr_best.pth", "cxr_ae.pth")]:
        if os.path.exists(new_name):
            os.remove(new_name)
        os.rename(old_name, new_name)
    
    print("\n训练完成！生成文件：ecg_ae.pth, pcg_ae.pth, cxr_ae.pth, thresholds.npz")
