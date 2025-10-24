"""
finetune_target.py
------------------
Fine-tune a pretrained encoder on the EpilepsyHAR target domain.
Automatically saves confusion matrix visualization to visualization/.
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


# ================================================================
# Dataset + Transform
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

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return torch.tensor(self.data[i], dtype=torch.float32), int(self.labels[i])


class TimeSeriesTransform:
    def __init__(self, normalize=True):
        self.normalize = normalize
        self.mean, self.std = None, None

    def fit(self, X):
        self.mean = X.mean((0, 1))
        self.std = X.std((0, 1)) + 1e-8

    def __call__(self, x):
        if self.normalize:
            x = (x - self.mean) / self.std
        return torch.tensor(x, dtype=torch.float32)


# ================================================================
# Encoder + Classifier
# ================================================================
class Encoder1D(nn.Module):
    def __init__(self, in_channels=9, emb_dim=256):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, 64, 5, padding=2),
            nn.ReLU(),
            nn.Conv1d(64, 128, 5, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.fc = nn.Linear(128, emb_dim)

    def forward(self, x):
        x = self.conv(x.permute(0, 2, 1)).squeeze(-1)
        return self.fc(x)


class ClassifierHead(nn.Module):
    def __init__(self, in_dim, n_classes):
        super().__init__()
        self.fc = nn.Linear(in_dim, n_classes)

    def forward(self, x):
        return self.fc(x)


# ================================================================
# Confusion Matrix Plotting
# ================================================================
def plot_confusion_matrix(y_true, y_pred, class_names, title="Confusion Matrix", out_dir="visualization"):
    os.makedirs(out_dir, exist_ok=True)
    cm = confusion_matrix(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=False,
        square=True,
        linewidths=0.5,
    )
    plt.title(f"{title}\nAccuracy={acc:.3f}, F1={f1:.3f}")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()

    save_path = os.path.join(out_dir, "confusion_matrix_finetune.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"💾 Saved confusion matrix → {save_path}")
    return cm


# ================================================================
# Fine-tuning Function
# ================================================================
def finetune_target(
    ckpt="runs/encoder_align_epoch10.pt",
    tgt_root="EpilepsyHAR_Realistic",
    batch_size=128,
    lr=1e-4,
    epochs=15,
):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---------- Load data ----------
    train = UCIHARDataset(tgt_root, "train")
    test = UCIHARDataset(tgt_root, "test")

    tfm = TimeSeriesTransform()
    tfm.fit(train.data)

    class Wrap(Dataset):
        def __init__(self, base, tfm):
            self.base, self.tfm = base, tfm

        def __len__(self):
            return len(self.base)

        def __getitem__(self, i):
            x, y = self.base[i]
            return self.tfm(x), y

    train_loader = DataLoader(Wrap(train, tfm), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(Wrap(test, tfm), batch_size=batch_size)

    # ---------- Load pretrained encoder ----------
    encoder = Encoder1D(in_channels=9, emb_dim=256).to(dev)
    cp = torch.load(ckpt, map_location=dev)
    encoder.load_state_dict(cp["encoder"], strict=False)
    encoder.eval()
    print(f"✅ Loaded encoder from {ckpt}")

    # Freeze encoder weights
    for p in encoder.parameters():
        p.requires_grad = False

    # ---------- Initialize classifier ----------
    n_classes = int(train.labels.max()) + 1
    head = ClassifierHead(256, n_classes).to(dev)
    opt = torch.optim.Adam(head.parameters(), lr=lr)

    # ---------- Training ----------
    for ep in range(1, epochs + 1):
        head.train()
        tot_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(dev), y.to(dev)
            with torch.no_grad():
                z = encoder(x)
            out = head(z)
            loss = F.cross_entropy(out, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot_loss += loss.item()
        print(f"[Epoch {ep:02d}] loss={tot_loss/len(train_loader):.4f}")

    # ---------- Evaluation ----------
    head.eval()
    ys, yh = [], []
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(dev), y.to(dev)
            z = encoder(x)
            p = head(z).argmax(1)
            ys += y.cpu().tolist()
            yh += p.cpu().tolist()

    acc = accuracy_score(ys, yh)
    f1 = f1_score(ys, yh, average="macro")
    print(f"✅ Target-domain accuracy={acc:.4f}, F1={f1:.4f}")

    # ---------- Confusion Matrix ----------
    class_names = ["Tonic-Clonic", "Myoclonic", "Absence", "Atonic"]
    cm = plot_confusion_matrix(ys, yh, class_names, "EpilepsyHAR Fine-tuned Model")

    return acc, f1, cm


# ================================================================
# CLI
# ================================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fine-tune on target domain")
    parser.add_argument("--ckpt", default="runs/encoder_align_epoch10.pt")
    parser.add_argument("--tgt_root", default="EpilepsyHAR_Realistic")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=15)
    args = parser.parse_args()

    finetune_target(args.ckpt, args.tgt_root, args.batch_size, args.lr, args.epochs)
# python finetune_target.py --ckpt runs/encoder_align_epoch10.pt
