"""
imu_diffusion_rq2.py

RQ2: Generative Modeling and Synthetic Data Realism for IMU-based pathological motion.
Exports data in full UCI-HAR folder structure (body_acc, body_gyro, total_acc, labels).
"""

import os
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------------------------------------
# Setup
# --------------------------------------------------
BASE_DIR = "Simulations"
TRAIN_DIR = os.path.join(BASE_DIR, "train")
TEST_DIR = os.path.join(BASE_DIR, "test")
for d in [TRAIN_DIR, TEST_DIR]:
    os.makedirs(d, exist_ok=True)

CLASSES = ["tremor", "myoclonic", "tonic_clonic", "freezing"]
CLASS2IDX = {c: i + 1 for i, c in enumerate(CLASSES)}  # 1-indexed like UCI HAR

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

# --------------------------------------------------
# Simulated physiological motion patterns
# --------------------------------------------------
def simulate_tremor(T, fs):
    t = np.arange(T) / fs
    f = np.random.uniform(3.5, 6.5)
    acc = 0.6 * np.sin(2 * np.pi * f * t) + 0.15 * np.random.randn(T)
    gyro = 0.4 * np.sin(2 * np.pi * f * t + np.pi/4) + 0.1 * np.random.randn(T)
    return acc, gyro


def simulate_myoclonic(T, fs):
    acc = 0.05 * np.random.randn(T)
    gyro = 0.03 * np.random.randn(T)
    for _ in range(np.random.randint(3, 6)):
        idx = np.random.randint(0, T - 5)
        width = np.random.randint(5, 15)
        amp = np.random.uniform(1, 2)
        slice_len = min(width, T - idx)
        win = np.hanning(width)[:slice_len]
        acc[idx:idx + slice_len] += amp * win
        gyro[idx:idx + slice_len] += 0.5 * amp * win
    return acc, gyro


def simulate_tonic_clonic(T, fs):
    t = np.arange(T) / fs
    split = T // 3
    tonic = 0.1 * np.tanh(5 * (t[:split] - t[split//2])) + 0.05 * np.random.randn(split)
    t2 = t[split:]
    f_t = np.linspace(2, 8, len(t2))
    clonic = 0.8 * np.sin(2*np.pi*f_t*t2) * (1 + 0.5*np.sin(2*np.pi*0.5*t2)) + 0.1*np.random.randn(len(t2))
    acc = np.concatenate([tonic, clonic])
    gyro = 0.6 * np.sin(2*np.pi*f_t*t2 + np.pi/6)
    gyro = np.concatenate([0.2*np.ones_like(tonic), gyro + 0.1*np.random.randn(len(t2))])
    return acc, gyro

def simulate_freezing(T, fs):
    acc = 0.02 * np.random.randn(T)
    gyro = 0.01 * np.random.randn(T)
    for _ in range(np.random.randint(3, 6)):
        start = np.random.randint(0, T-30)
        width = np.random.randint(20, 50)
        micro = 0.15 * np.sin(2*np.pi*np.random.uniform(5, 9)*np.arange(width)/fs)
        acc[start:start+width] += micro[:min(width, T-start)]
        gyro[start:start+width] += 0.5 * micro[:min(width, T-start)]
    return acc, gyro

SIM_FUNCS = {
    "tremor": simulate_tremor,
    "myoclonic": simulate_myoclonic,
    "tonic_clonic": simulate_tonic_clonic,
    "freezing": simulate_freezing,
}

# --------------------------------------------------
# Dataset Simulation
# --------------------------------------------------
def simulate_dataset(n_per_class=200, T=256, fs=50.0):
    data = []
    labels = []
    for cname, func in SIM_FUNCS.items():
        for _ in range(n_per_class):
            acc_x, gyro_x = func(T, fs)
            acc_y, gyro_y = func(T, fs)
            acc_z, gyro_z = func(T, fs)
            # total_acc = body_acc + small gravity + random offset
            total_x = acc_x + 0.1*np.random.randn(T)
            total_y = acc_y + 0.1*np.random.randn(T)
            total_z = acc_z + 0.1*np.random.randn(T)
            data.append([acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z, total_x, total_y, total_z])
            labels.append(CLASS2IDX[cname])
    data = np.array(data)  # [N, 9, T]
    labels = np.array(labels)
    return data, labels

# --------------------------------------------------
# Save Functions (UCI HAR Format)
# --------------------------------------------------
def save_ucihar_split(folder, X, y, prefix):
    sensors = [
        "body_acc_x", "body_acc_y", "body_acc_z",
        "body_gyro_x", "body_gyro_y", "body_gyro_z",
        "total_acc_x", "total_acc_y", "total_acc_z"
    ]
    for i, s in enumerate(sensors):
        fname = f"{s}_{prefix}.txt"
        fpath = os.path.join(folder, fname)
        np.savetxt(fpath, X[:, i, :])
        print(f"[saved] {fpath} ({X.shape[0]} samples)")
    # Save labels
    ypath = os.path.join(folder, f"y_{prefix}.txt")
    np.savetxt(ypath, y, fmt="%d")
    print(f"[saved] {ypath}")

# --------------------------------------------------
# (Optional) Simple Diffusion Model
# --------------------------------------------------
class TinyUNet1D(nn.Module):
    def __init__(self, C=9, base=32, n_classes=4):
        super().__init__()
        self.emb = nn.Embedding(n_classes, base)
        self.conv1 = nn.Conv1d(C, base, 3, padding=1)
        self.conv2 = nn.Conv1d(base, base, 3, padding=1)
        self.out = nn.Conv1d(base, C, 3, padding=1)
    def forward(self, x, y):
        e = self.emb(y)[:, :, None]
        h = F.silu(self.conv1(x)) + e
        h = F.silu(self.conv2(h))
        return self.out(h)

# def train_toy_diffusion(X, y, epochs=2):
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model = TinyUNet1D(C=9, n_classes=len(np.unique(y))).to(device)
#     opt = torch.optim.Adam(model.parameters(), lr=1e-3)
#     X = torch.tensor(X).to(device).float()
#     y = torch.tensor(y-1).to(device)  # make 0-based
#     X = X.permute(0, 2, 1)  # [N, T, C] -> [N, C, T]
#     for ep in range(epochs):
#         perm = torch.randperm(len(X))
#         for i in range(0, len(X), 32):
#             idx = perm[i:i+32]
#             xb, yb = X[idx], y[idx]
#             noise = torch.randn_like(xb)
#             pred = model(xb + 0.1*noise, yb)
#             loss = F.mse_loss(pred, xb)
#             opt.zero_grad(); loss.backward(); opt.step()
#         print(f"[Diffusion] epoch {ep+1}/{epochs} loss={loss.item():.4f}")
#     torch.save(model.state_dict(), os.path.join(BASE_DIR, "ddpm_latest.pt"))
#     return model
def train_toy_diffusion(X, y, epochs=2):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TinyUNet1D(C=9, n_classes=len(np.unique(y))).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # Prepare tensors
    X = torch.tensor(X).float().to(device)        # [N, 9, T]
    y = torch.tensor(y - 1).long().to(device)     # make 0-based for embedding
    X = X.permute(0, 1, 2)                        # ensure [N, C, T]
    
    for ep in range(epochs):
        perm = torch.randperm(len(X))
        for i in range(0, len(X), 32):
            idx = perm[i:i+32]
            xb, yb = X[idx], y[idx]
            noise = torch.randn_like(xb)
            pred = model(xb + 0.1 * noise, yb)
            loss = F.mse_loss(pred, xb)
            opt.zero_grad()
            loss.backward()
            opt.step()
        print(f"[Diffusion] epoch {ep+1}/{epochs} loss={loss.item():.4f}")
    
    torch.save(model.state_dict(), os.path.join(BASE_DIR, "ddpm_latest.pt"))
    return model
# --------------------------------------------------
# Main Script
# --------------------------------------------------
if __name__ == "__main__":
    set_seed(42)
    print("[1] Simulating IMU dataset...")
    X, y = simulate_dataset(200, 256, 50)

    print("[2] Splitting into train/test...")
    idx = np.arange(len(X))
    np.random.shuffle(idx)
    split = int(0.8 * len(X))
    tr_idx, te_idx = idx[:split], idx[split:]
    X_train, y_train = X[tr_idx], y[tr_idx]
    X_test, y_test = X[te_idx], y[te_idx]

    print("[3] Saving to UCI-HAR format...")
    save_ucihar_split(TRAIN_DIR, X_train, y_train, "train")
    save_ucihar_split(TEST_DIR, X_test, y_test, "test")

    print("[4] (Optional) Training toy diffusion model...")
    _ = train_toy_diffusion(X_train, y_train, epochs=2)

    print(f"\nAll data saved under: {BASE_DIR}/")
# python imu_diffusion_rq2.py