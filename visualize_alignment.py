"""
visualize_alignment.py
----------------------
Visualize learned feature alignment between source (UCI HAR)
and target (EpilepsyHAR) domains using a trained encoder checkpoint.
"""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

# ================================================================
# Utility functions
# ================================================================
def device_autoselect():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ================================================================
# Dataset and transform
# ================================================================
class UCIHARDataset(Dataset):
    def __init__(self, root, split="train"):
        signals_dir = os.path.join(root, split, "Inertial Signals")
        y_path = os.path.join(root, split, f"y_{split}.txt")
        self.labels = np.loadtxt(y_path).astype(int)
        signal_files = sorted([f for f in os.listdir(signals_dir) if f.endswith(".txt")])
        X = [np.loadtxt(os.path.join(signals_dir, f)) for f in signal_files]
        self.data = np.stack(X, axis=-1)
        if self.labels.min() != 0:
            self.labels -= self.labels.min()
    def __len__(self): return len(self.labels)
    def __getitem__(self, i): return torch.tensor(self.data[i], dtype=torch.float32), int(self.labels[i])

class TimeSeriesTransform:
    def __init__(self, normalize=True):
        self.normalize = normalize
        self.mean, self.std = None, None
    def fit(self, X):
        self.mean = np.mean(X, axis=(0,1))
        self.std = np.std(X, axis=(0,1)) + 1e-8
    def __call__(self, x):
        if self.normalize:
            x = (x - self.mean) / self.std
        return torch.tensor(x, dtype=torch.float32)

# ================================================================
# Model definition
# ================================================================
import torch.nn as nn
import torch.nn.functional as F

class Encoder1D(nn.Module):
    def __init__(self, in_channels=9, emb_dim=256, use_lstm=False):
        super().__init__()
        self.use_lstm = use_lstm
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, 64, 5, padding=2), nn.ReLU(),
            nn.Conv1d(64, 128, 5, padding=2), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        self.fc = nn.Linear(128, emb_dim)
        if use_lstm:
            self.lstm = nn.LSTM(in_channels, 64, batch_first=True, bidirectional=True)
            self.fc = nn.Linear(128, emb_dim)
    def forward(self, x):
        if self.use_lstm:
            out, _ = self.lstm(x)
            out = out.mean(1)
        else:
            out = self.conv(x.permute(0,2,1)).squeeze(-1)
        return self.fc(out)

# ================================================================
# Visualization function
# ================================================================
def visualize_alignment(
    ckpt_path="runs/encoder_align_epoch10.pt",
    uci_root="UCIHARDataset",
    tgt_root="EpilepsyHAR_Realistic",
    batch_size=256,
    method="tsne",      # or "pca"
    save_fig=True
):
    dev = device_autoselect()

    # ---- load encoder ----
    encoder = Encoder1D(in_channels=9, emb_dim=256).to(dev)
    cp = torch.load(ckpt_path, map_location=dev)
    encoder.load_state_dict(cp["encoder"], strict=False)
    encoder.eval()
    print(f"✅ Loaded encoder from {ckpt_path}")

    # ---- load datasets ----
    src_ds = UCIHARDataset(uci_root, "train")
    tgt_ds = UCIHARDataset(tgt_root, "train")

    tfm_src = TimeSeriesTransform(normalize=True); tfm_src.fit(src_ds.data)
    class Wrap(Dataset):
        def __init__(self, base, tfm): self.base, self.tfm = base, tfm
        def __len__(self): return len(self.base)
        def __getitem__(self, i): x,y=self.base[i]; return self.tfm(x), y

    src_loader = DataLoader(Wrap(src_ds, tfm_src), batch_size=batch_size)
    tgt_loader = DataLoader(Wrap(tgt_ds, tfm_src), batch_size=batch_size)

    # ---- compute embeddings ----
    zs, zt = [], []
    with torch.no_grad():
        for x,_ in src_loader:
            zs.append(encoder(x.to(dev)).cpu())
        for x,_ in tgt_loader:
            zt.append(encoder(x.to(dev)).cpu())
    zs, zt = torch.cat(zs), torch.cat(zt)
    emb = torch.cat([zs, zt])
    domain_labels = np.array(["Source"]*len(zs) + ["Target"]*len(zt))
    print(f"Embeddings: source {zs.shape}, target {zt.shape}")

    # ---- dimensionality reduction ----
    if method.lower() == "tsne":
        print("Running t-SNE... this may take a few minutes.")
        reducer = TSNE(n_components=2, init="random", learning_rate="auto", perplexity=30)
    else:
        reducer = PCA(n_components=2)
    emb_2d = reducer.fit_transform(emb)

    # ---- plot ----
    plt.figure(figsize=(7,6))
    plt.scatter(emb_2d[:len(zs),0], emb_2d[:len(zs),1], s=10, alpha=0.5, label="UCI HAR (Source)")
    plt.scatter(emb_2d[len(zs):,0], emb_2d[len(zs):,1], s=10, alpha=0.5, label="EpilepsyHAR (Target)")
    plt.legend()
    plt.title(f"Domain Alignment Visualization ({method.upper()})")
    plt.xlabel(f"{method.upper()} dim 1"); plt.ylabel(f"{method.upper()} dim 2")
    plt.tight_layout()
    if save_fig:
        out_path = os.path.join(os.path.dirname(ckpt_path), f"alignment_{method}.png")
        plt.savefig(out_path, dpi=300)
        print(f"💾 Saved figure to {out_path}")
    plt.show()

# ================================================================
# CLI entry
# ================================================================
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Visualize domain alignment results")
    parser.add_argument("--ckpt", default="runs/encoder_align_epoch10.pt")
    parser.add_argument("--uci_root", default="UCIHARDataset")
    parser.add_argument("--tgt_root", default="EpilepsyHAR_Realistic")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--method", default="tsne", choices=["tsne", "pca"])
    parser.add_argument("--no_save", action="store_true", help="Do not save figure")
    args = parser.parse_args()

    visualize_alignment(
        ckpt_path=args.ckpt,
        uci_root=args.uci_root,
        tgt_root=args.tgt_root,
        batch_size=args.batch_size,
        method=args.method,
        save_fig=not args.no_save
    )
