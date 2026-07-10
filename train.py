import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from model import ECGAutoencoder, PCGAutoencoder, CXRAutoencoder
from tqdm import tqdm
import config


class SingleModalDataset(Dataset):
    def __init__(self, data_path, name):
        self.name = name
        self.data = np.load(data_path, mmap_mode="r")
        print(f"[{self.name}] using {len(self.data)} samples from {data_path}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = torch.tensor(self.data[idx], dtype=torch.float32)
        if self.name == "cxr" and x.ndim == 2:
            x = x.unsqueeze(0)
        return x


def build_loaders(data_path, name, batch_size, val_size=0.2, random_state=42):
    dataset = SingleModalDataset(data_path, name)
    indices = np.arange(len(dataset))
    rng = np.random.default_rng(random_state)
    rng.shuffle(indices)
    val_count = max(1, int(np.ceil(len(indices) * val_size)))
    val_indices = indices[:val_count]
    train_indices = indices[val_count:]
    train_loader = DataLoader(
        Subset(dataset, train_indices),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        Subset(dataset, val_indices),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    print(f"[{name}] train={len(train_indices)}, val={len(val_indices)}")
    return train_loader, val_loader


def train_one_model(model, loader, criterion, optimizer, device, epochs, name):
    model.train()
    best_loss = float("inf")
    for epoch in range(epochs):
        total_loss = 0.0
        for x in tqdm(loader, desc=f"{name} Epoch {epoch + 1}"):
            x = x.to(device)
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
        for x in loader:
            x = x.to(device)
            recon = model(x)
            if name == "cxr":
                err = torch.mean((recon - x) ** 2, dim=(1, 2, 3))
            else:
                err = torch.mean((recon - x) ** 2, dim=(1, 2))
            errors.extend(err.cpu().numpy())
    thresh = np.percentile(errors, 95)
    print(f"[{name}] threshold (95%): {thresh:.6f}")
    return thresh


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=config.EPOCHS)
    parser.add_argument("--batch_size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=config.LEARNING_RATE)
    args = parser.parse_args()

    device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    ecg_train_loader, ecg_val_loader = build_loaders(
        os.path.join(config.OUTPUT_DIR, "normal_ecg.npy"),
        "ecg",
        args.batch_size,
    )
    pcg_train_loader, pcg_val_loader = build_loaders(
        os.path.join(config.OUTPUT_DIR, "normal_pcg.npy"),
        "pcg",
        args.batch_size,
    )
    cxr_train_loader, cxr_val_loader = build_loaders(
        os.path.join(config.OUTPUT_DIR, "normal_cxr.npy"),
        "cxr",
        args.batch_size,
    )

    criterion = nn.HuberLoss(delta=1.0)

    print("\nTraining ECG Autoencoder")
    ecg_model = ECGAutoencoder().to(device)
    ecg_model = train_one_model(
        ecg_model,
        ecg_train_loader,
        criterion,
        optim.Adam(ecg_model.parameters(), lr=args.lr),
        device,
        args.epochs,
        "ecg",
    )

    print("\nTraining PCG Autoencoder")
    pcg_model = PCGAutoencoder().to(device)
    pcg_model = train_one_model(
        pcg_model,
        pcg_train_loader,
        criterion,
        optim.Adam(pcg_model.parameters(), lr=args.lr),
        device,
        args.epochs,
        "pcg",
    )

    print("\nTraining CXR Autoencoder")
    cxr_model = CXRAutoencoder().to(device)
    cxr_model = train_one_model(
        cxr_model,
        cxr_train_loader,
        criterion,
        optim.Adam(cxr_model.parameters(), lr=args.lr),
        device,
        args.epochs,
        "cxr",
    )

    ecg_model.load_state_dict(torch.load("ecg_best.pth", map_location=device, weights_only=True))
    pcg_model.load_state_dict(torch.load("pcg_best.pth", map_location=device, weights_only=True))
    cxr_model.load_state_dict(torch.load("cxr_best.pth", map_location=device, weights_only=True))

    ecg_thresh = compute_threshold(ecg_model, ecg_val_loader, device, "ecg")
    pcg_thresh = compute_threshold(pcg_model, pcg_val_loader, device, "pcg")
    cxr_thresh = compute_threshold(cxr_model, cxr_val_loader, device, "cxr")

    np.savez("thresholds.npz", ecg=ecg_thresh, pcg=pcg_thresh, cxr=cxr_thresh)

    for old_name, new_name in [
        ("ecg_best.pth", "ecg_ae.pth"),
        ("pcg_best.pth", "pcg_ae.pth"),
        ("cxr_best.pth", "cxr_ae.pth"),
    ]:
        if os.path.exists(new_name):
            os.remove(new_name)
        os.rename(old_name, new_name)

    print("\nTraining complete: ecg_ae.pth, pcg_ae.pth, cxr_ae.pth, thresholds.npz")
