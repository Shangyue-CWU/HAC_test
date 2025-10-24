"""
plot_alignment_evolution.py
---------------------------
Visualize how domain alignment metrics evolve over training epochs.
Scans a folder (e.g. runs/) for encoder checkpoints and computes
MMD, Cosine, and Proxy A-distance for each, then saves plots to visualization/.
"""

import os, re, torch, numpy as np
import matplotlib.pyplot as plt
from evaluate_alignment_metrics import evaluate_alignment   # Step 4.2 script

def extract_epoch_num(fname):
    m = re.search(r"epoch[_-]?(\d+)", fname)
    return int(m.group(1)) if m else None

def plot_metric_trends(results, out_dir="visualization"):
    os.makedirs(out_dir, exist_ok=True)

    epochs = sorted(results.keys())
    metrics = ["MMD", "Cosine", "A_distance"]

    plt.figure(figsize=(7,5))
    for m in metrics:
        plt.plot(epochs, [results[e][m] for e in epochs], marker="o", label=m)
    plt.xlabel("Epoch")
    plt.ylabel("Metric Value")
    plt.title("Domain Alignment Metric Evolution")
    plt.legend()
    plt.grid(True)
    path = os.path.join(out_dir, "alignment_metrics_evolution.png")
    plt.tight_layout(); plt.savefig(path, dpi=300)
    print(f"Saved plot → {path}")
    plt.show()

def visualize_metric_evolution(run_dir="runs",
                               uci_root="UCIHARDataset",
                               tgt_root="EpilepsyHAR_Realistic",
                               batch_size=256):
    checkpoints = [os.path.join(run_dir,f) for f in os.listdir(run_dir)
                   if f.endswith(".pt") and "encoder" in f]
    checkpoints = sorted(checkpoints, key=extract_epoch_num)
    if not checkpoints:
        print(f"No checkpoints found in {run_dir}")
        return

    results = {}
    for ckpt in checkpoints:
        ep = extract_epoch_num(ckpt)
        if ep is None: continue
        print(f"\nEvaluating epoch {ep} ...")
        res = evaluate_alignment(ckpt, uci_root, tgt_root, batch_size)
        results[ep] = res

    plot_metric_trends(results, "visualization")
    np.save(os.path.join("visualization", "alignment_metrics.npy"), results)
    print("Saved numeric results to visualization/alignment_metrics.npy")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Plot metric evolution over epochs")
    parser.add_argument("--run_dir", default="runs")
    parser.add_argument("--uci_root", default="UCIHARDataset")
    parser.add_argument("--tgt_root", default="EpilepsyHAR_Realistic")
    parser.add_argument("--batch_size", type=int, default=256)
    args = parser.parse_args()
    visualize_metric_evolution(args.run_dir, args.uci_root, args.tgt_root, args.batch_size)
    
    # python plot_alignment_evolution.py --run_dir runs

