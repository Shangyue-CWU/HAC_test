"""
evaluate_alignment_metrics.py
-----------------------------
Quantitatively evaluate cross-domain feature alignment between
source (UCI HAR) and target (EpilepsyHAR) datasets.
"""
# This step lets you numerically confirm that your domain adaptation (contrastive + CORAL) actually aligned UCI HAR (source) and EpilepsyHAR (target) feature spaces, even before fine-tuning.
# We’ll measure how close the feature distributions of the two domains are by computing:
# | Metric                             | Meaning                                             | Expected behavior (better alignment ↓)  |
# | ---------------------------------- | --------------------------------------------------- | --------------------------------------- |
# | **MMD (Maximum Mean Discrepancy)** | Distance between domain feature means + covariances | Smaller = better                        |
# | **Cosine Distance**                | Average 1 − cos similarity between features         | Smaller = better                        |
# | **A-distance (Proxy 𝒜-distance)** | Error of a domain discriminator (binary classifier) | Closer to 0 = indistinguishable domains |


import os, numpy as np, torch
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import torch.nn as nn
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist

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
    def __len__(self): return len(self.labels)
    def __getitem__(self, i): return torch.tensor(self.data[i], dtype=torch.float32), int(self.labels[i])

class TimeSeriesTransform:
    def __init__(self, normalize=True):
        self.normalize = normalize; self.mean=None; self.std=None
    def fit(self,X): self.mean=X.mean((0,1)); self.std=X.std((0,1))+1e-8
    def __call__(self,x):
        if self.normalize: x=(x-self.mean)/self.std
        return torch.tensor(x,dtype=torch.float32)

# ================================================================
# Encoder definition (same as training)
# ================================================================
class Encoder1D(nn.Module):
    def __init__(self,in_channels=9,emb_dim=256):
        super().__init__()
        self.conv=nn.Sequential(
            nn.Conv1d(in_channels,64,5,padding=2),nn.ReLU(),
            nn.Conv1d(64,128,5,padding=2),nn.ReLU(),
            nn.AdaptiveAvgPool1d(1))
        self.fc=nn.Linear(128,emb_dim)
    def forward(self,x):
        x=self.conv(x.permute(0,2,1)).squeeze(-1)
        return self.fc(x)

# ================================================================
# Metric helpers
# ================================================================
# def mmd_loss(X,Y,kernel='rbf',sigma=1.0):
#     """Compute Maximum Mean Discrepancy between two sets of features"""
#     XX = torch.mm(X, X.t())
#     YY = torch.mm(Y, Y.t())
#     XY = torch.mm(X, Y.t())
#     rx = (XX.diag().unsqueeze(0).expand_as(XX))
#     ry = (YY.diag().unsqueeze(0).expand_as(YY))
#     Kxx = torch.exp(- (rx.t() + rx - 2*XX) / (2*sigma**2))
#     Kyy = torch.exp(- (ry.t() + ry - 2*YY) / (2*sigma**2))
#     Kxy = torch.exp(- (rx.t() + ry - 2*XY) / (2*sigma**2))
#     return Kxx.mean() + Kyy.mean() - 2*Kxy.mean()
def mmd_loss(X, Y, sigma=1.0):
    """
    Compute unbiased Maximum Mean Discrepancy (MMD^2) between feature sets X and Y.
    Works even if X and Y have different sample counts.
    """
    n, m = X.size(0), Y.size(0)
    XX = torch.cdist(X, X, p=2) ** 2
    YY = torch.cdist(Y, Y, p=2) ** 2
    XY = torch.cdist(X, Y, p=2) ** 2

    Kxx = torch.exp(-XX / (2 * sigma ** 2))
    Kyy = torch.exp(-YY / (2 * sigma ** 2))
    Kxy = torch.exp(-XY / (2 * sigma ** 2))

    mmd = Kxx.mean() + Kyy.mean() - 2 * Kxy.mean()
    return mmd

def cosine_distance(X,Y):
    Xn = F.normalize(X,dim=1)
    Yn = F.normalize(Y,dim=1)
    sim = torch.mm(Xn,Yn.t())
    return 1 - sim.mean()

def proxy_a_distance(src_emb, tgt_emb):
    """Train a domain classifier and compute A-distance = 2(1-2ε)"""
    X = np.vstack([src_emb, tgt_emb])
    y = np.hstack([np.zeros(len(src_emb)), np.ones(len(tgt_emb))])
    Xtr,Xte,ytr,yte = train_test_split(X,y,test_size=0.3,random_state=42)
    sc = StandardScaler().fit(Xtr)
    Xtr,Xte = sc.transform(Xtr), sc.transform(Xte)
    clf = LinearSVC(max_iter=5000)
    clf.fit(Xtr,ytr)
    acc = accuracy_score(yte, clf.predict(Xte))
    a_dist = 2 * (1 - 2 * acc)
    return a_dist, acc

# ================================================================
# Main evaluation function
# ================================================================
def evaluate_alignment(ckpt="runs/encoder_align_epoch10.pt",
                       uci_root="UCIHARDataset",
                       tgt_root="EpilepsyHAR_Realistic",
                       batch_size=256):
    dev=torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load encoder
    encoder=Encoder1D(in_channels=9,emb_dim=256).to(dev)
    cp=torch.load(ckpt,map_location=dev)
    encoder.load_state_dict(cp["encoder"],strict=False)
    encoder.eval()
    print("✅ Loaded encoder", ckpt)

    # Load data
    src=UCIHARDataset(uci_root,"train")
    tgt=UCIHARDataset(tgt_root,"train")
    tfm=TimeSeriesTransform(); tfm.fit(src.data)
    class Wrap(Dataset):
        def __init__(self,b,t): self.b,self.t=b,t
        def __len__(self): return len(self.b)
        def __getitem__(self,i): x,y=self.b[i]; return self.t(x),y
    src_loader=DataLoader(Wrap(src,tfm),batch_size=batch_size)
    tgt_loader=DataLoader(Wrap(tgt,tfm),batch_size=batch_size)

    zs,zt=[],[]
    with torch.no_grad():
        for x,_ in src_loader: zs.append(encoder(x.to(dev)).cpu())
        for x,_ in tgt_loader: zt.append(encoder(x.to(dev)).cpu())
    zs,zt=torch.cat(zs),torch.cat(zt)
    print(f"Features: src={zs.shape}, tgt={zt.shape}")

    # -------------- Compute metrics --------------
    mmd = mmd_loss(zs,zt).item()
    cos = cosine_distance(zs,zt).item()
    a_dist, dom_acc = proxy_a_distance(zs.numpy(), zt.numpy())

    print("\n🔹 Domain Alignment Metrics:")
    print(f"  • MMD Distance        : {mmd:.4f}")
    print(f"  • Cosine Distance     : {cos:.4f}")
    print(f"  • Proxy A-distance    : {a_dist:.4f} (domain clf acc={dom_acc:.3f})")

    return {"MMD":mmd, "Cosine":cos, "A_distance":a_dist, "DomainAcc":dom_acc}

# ================================================================
# CLI
# ================================================================
if __name__=="__main__":
    import argparse
    parser=argparse.ArgumentParser(description="Evaluate domain alignment metrics")
    parser.add_argument("--ckpt",default="runs/encoder_align_epoch10.pt")
    parser.add_argument("--uci_root",default="UCIHARDataset")
    parser.add_argument("--tgt_root",default="EpilepsyHAR_Realistic")
    parser.add_argument("--batch_size",type=int,default=256)
    args=parser.parse_args()
    evaluate_alignment(args.ckpt,args.uci_root,args.tgt_root,args.batch_size)
    
    
# python evaluate_alignment_metrics.py --ckpt runs/encoder_align_epoch10.pt

