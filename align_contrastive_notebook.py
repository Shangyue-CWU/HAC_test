"""
align_contrastive_notebook_fixed.py
-----------------------------------
Cross-domain contrastive alignment pipeline for
"Detecting Pathological Human Motion from Daily IMU Data".

Features:
- UCI HAR -> EpilepsyHAR transfer
- Source supervised + target contrastive + CORAL
- Automatic checkpoint saving / resume
- CLI arguments
"""

import os, math, random, time, argparse, numpy as np
from collections import Counter
from datetime import datetime
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn

# ================================================================
# Utility functions
# ================================================================
def set_seed_safe(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        try:
            torch.cuda.manual_seed_all(seed)
        except Exception as e:
            print("⚠️ Skipped cuda.manual_seed_all:", repr(e))

def device_autoselect():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def ensure_dir(path):
    if not os.path.exists(path): os.makedirs(path)

# ================================================================
# Dataset classes
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
    def __init__(self, normalize=True, augment=False):
        self.normalize, self.augment = normalize, augment
        self.mean, self.std = None, None
    def fit(self, X):
        self.mean = np.mean(X, axis=(0,1))
        self.std = np.std(X, axis=(0,1)) + 1e-8
    def __call__(self, x):
        if self.normalize:
            x = (x - self.mean) / self.std
        return torch.tensor(x, dtype=torch.float32)

# ================================================================
# Model definitions
# ================================================================
class Encoder1D(nn.Module):
    def __init__(self, in_channels=9, emb_dim=128, use_lstm=False):
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

class ClassifierHead(nn.Module):
    def __init__(self, in_dim, n_classes):
        super().__init__()
        self.fc = nn.Linear(in_dim, n_classes)
    def forward(self, x): return self.fc(x)

def coral_loss(source, target):
    def cov(feat):
        xm = feat - feat.mean(0, keepdim=True)
        return (xm.t() @ xm) / (feat.size(0)-1)
    cs, ct = cov(source), cov(target)
    return F.mse_loss(cs, ct)

def domain_confusion_score(encoder, src_loader, tgt_loader, dev):
    encoder.eval()
    zs, zt = [], []
    with torch.no_grad():
        for x,_ in src_loader: zs.append(encoder(x.to(dev)))
        for x,_ in tgt_loader: zt.append(encoder(x.to(dev)))
    zs, zt = torch.cat(zs), torch.cat(zt)
    d = torch.norm(zs.mean(0)-zt.mean(0)).item()
    return float(np.exp(-d))

def evaluate(encoder, head, loader, dev):
    encoder.eval(); head.eval()
    all_y, all_p = [], []
    with torch.no_grad():
        for x,y in loader:
            x,y = x.to(dev), y.to(dev)
            z = encoder(x)
            p = head(z).argmax(1).cpu()
            all_y.append(y.cpu()); all_p.append(p)
    y = torch.cat(all_y); p = torch.cat(all_p)
    acc = (p==y).float().mean().item()
    f1 = 2*(acc*acc)/(acc+acc+1e-8)
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y, p)
    return acc, f1, cm

# ================================================================
# Main alignment pipeline
# ================================================================
def align_contrastive_train(args):
    set_seed_safe(args.seed)
    dev = device_autoselect()
   

    # Load datasets
    src_tr = UCIHARDataset(args.uci_root, "train")
    src_te = UCIHARDataset(args.uci_root, "test")
    tgt_tr = UCIHARDataset(args.tgt_root, "train")
    tgt_te = UCIHARDataset(args.tgt_root, "test")

    src_classes = int(np.max(src_tr.labels)) + 1
    tgt_classes = int(np.max(tgt_tr.labels)) + 1
    print(f"Source classes={src_classes} | Target classes={tgt_classes}")

    tfm_src = TimeSeriesTransform(normalize=True); tfm_src.fit(src_tr.data)

    class Wrap(Dataset):
        def __init__(self, base, tfm): self.base, self.tfm = base, tfm
        def __len__(self): return len(self.base)
        def __getitem__(self, i): x,y = self.base[i]; return self.tfm(x), y

    src_loader = DataLoader(Wrap(src_tr, tfm_src), batch_size=args.batch_size, shuffle=True, drop_last=True)
    tgt_loader = DataLoader(Wrap(tgt_tr, tfm_src), batch_size=args.batch_size, shuffle=True, drop_last=True)
    src_val    = DataLoader(Wrap(src_te, tfm_src), batch_size=args.batch_size)
    tgt_val    = DataLoader(Wrap(tgt_te, tfm_src), batch_size=args.batch_size)

    # Model
    in_ch = src_tr[0][0].shape[1]
    encoder = Encoder1D(in_channels=in_ch, emb_dim=args.emb_dim).to(dev)
    head_src = ClassifierHead(args.emb_dim, src_classes).to(dev)

    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=dev)
        encoder.load_state_dict(ckpt["encoder"], strict=False)
        print(f"🔹 Resumed from {args.resume}")
    elif os.path.exists(args.ckpt):
        try:
            cp = torch.load(args.ckpt, map_location=dev)
            encoder.load_state_dict(cp["encoder"], strict=False)
            print("🔹 Loaded pretrained encoder.")
        except Exception as e:
            print("⚠️ Could not load pretrained:", e)

    opt = torch.optim.AdamW(list(encoder.parameters())+list(head_src.parameters()), lr=args.lr, weight_decay=args.weight_decay)

    # Target prototypes
    K = tgt_classes
    proto_tgt = torch.zeros(K, args.emb_dim, device=dev)
    cnt_tgt   = torch.zeros(K, device=dev)
    @torch.no_grad()
    def update_proto(z, y):
        for k in range(K):
            m = (y==k)
            if m.any():
                n = m.sum()
                proto_tgt[k] = (proto_tgt[k]*cnt_tgt[k] + z[m].mean(0)*n) / (cnt_tgt[k]+n)
                cnt_tgt[k] += n

    # Initialize prototypes
    encoder.eval()
    with torch.no_grad():
        for x,y in DataLoader(Wrap(tgt_tr, tfm_src), batch_size=args.batch_size):
            z = encoder(x.to(dev)); update_proto(z, y.to(dev))

    def eval_probe():
        encoder.eval()
        head_tgt = ClassifierHead(args.emb_dim, tgt_classes).to(dev)
        opt_probe = torch.optim.Adam(head_tgt.parameters(), lr=1e-3)
        tr = DataLoader(Wrap(tgt_tr, tfm_src), batch_size=args.batch_size, shuffle=True)
        for _ in range(args.probe_epochs):
            head_tgt.train()
            for x,y in tr:
                x,y = x.to(dev), y.to(dev)
                with torch.no_grad(): z = encoder(x)
                l = F.cross_entropy(head_tgt(z), y)
                opt_probe.zero_grad(); l.backward(); opt_probe.step()
        return evaluate(encoder, head_tgt, tgt_val, dev)

    # Training loop
    for ep in range(1, args.epochs+1):
        encoder.train(); head_src.train()
        tot = tot_cls = tot_align = tot_coral = 0.0
        for (xs, ys), (xt, yt) in zip(src_loader, tgt_loader):
            xs, ys, xt, yt = xs.to(dev), ys.to(dev, dtype=torch.long), xt.to(dev), yt.to(dev, dtype=torch.long)
            zs, zt = encoder(xs), encoder(xt)
            loss = 0.0

            if args.supervise_source:
                logits_s = head_src(zs)
                if ys.min() < 0 or ys.max() >= logits_s.shape[1]:
                    raise RuntimeError(f"Bad source label range {ys.min()}..{ys.max()} vs {logits_s.shape[1]}")
                cls = F.cross_entropy(logits_s, ys)
                loss += args.lambda_src * cls; tot_cls += cls.item()

            protos = F.normalize(proto_tgt.detach(), dim=-1)
            sim = torch.mm(F.normalize(zt, dim=-1), protos.t()) / args.temperature
            if yt.min() < 0 or yt.max() >= K:
                raise RuntimeError(f"Bad target label range {yt.min()}..{yt.max()} vs {K}")
            align = F.cross_entropy(sim, yt)
            loss += align; tot_align += align.item()

            if args.use_coral:
                cr = coral_loss(zs, zt); loss += args.lambda_coral * cr; tot_coral += cr.item()

            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            with torch.no_grad(): update_proto(zt, yt)
            tot += loss.item()

        acc, f1, cm = eval_probe()
        dom_acc = domain_confusion_score(encoder, src_val, tgt_val, dev)
        print(f"[E{ep:02d}] loss={tot/len(src_loader):.4f} cls={tot_cls/len(src_loader):.4f} "
              f"align={tot_align/len(src_loader):.4f} coral={tot_coral/len(src_loader):.4f} | "
              f"tgt acc={acc:.4f} f1={f1:.4f} | dom_acc={dom_acc:.3f}")
        print("CM:\n", cm)

        # Save checkpoint each epoch
        ckpt_path = os.path.join(f"runs/encoder_align_epoch{ep}.pt")
        torch.save({"encoder": encoder.state_dict()}, ckpt_path)
        print(f"Saved checkpoint: {ckpt_path}")

    print("Contrastive alignment training completed.")

# ================================================================
# CLI Entrypoint
# ================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cross-domain alignment trainer")
    parser.add_argument("--uci_root", default="UCIHARDataset")
    parser.add_argument("--tgt_root", default="EpilepsyHAR_Realistic")
    parser.add_argument("--ckpt", default="runs/simclr_src_last.pt")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--emb_dim", type=int, default=256)
    parser.add_argument("--lambda_src", type=float, default=0.5)
    parser.add_argument("--lambda_coral", type=float, default=0.1)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--probe_epochs", type=int, default=3)
    parser.add_argument("--supervise_source", action="store_true")
    parser.add_argument("--use_coral", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    align_contrastive_train(args)
