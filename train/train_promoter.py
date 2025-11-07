import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, roc_curve, auc
from sklearn.model_selection import train_test_split
from random import sample
import matplotlib.pyplot as plt
from tqdm import tqdm

from utils.config import *               # 경로·파라미터 설정
from models.encoder import HMTransformerTokenEncoder  # Encoder 구조 재사용


# ───────────────────────────────────────────────────────────────
# Dataset
# ───────────────────────────────────────────────────────────────
class PromoterDataset(Dataset):
    """Promoter vs Non-promoter binary classification dataset."""
    def __init__(self, matrix, samples):
        self.matrix = matrix.astype(np.float32)
        self.samples = samples

    def __len__(self): 
        return len(self.samples)

    def __getitem__(self, idx):
        center, label = self.samples[idx]
        win = self.matrix[center - WINDOW_SIZE//2 : center + WINDOW_SIZE//2 + 1]
        return torch.from_numpy(win), torch.tensor(label, dtype=torch.float32)


# ───────────────────────────────────────────────────────────────
# Classifier
# ───────────────────────────────────────────────────────────────
class EpiPRClassifier(nn.Module):
    """Transformer encoder + dual pooling classifier."""
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        self.dropout = nn.Dropout(0.3)
        self.mlp = nn.Sequential(
            nn.Linear(64 * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        z = self.encoder(x)
        z_mean = z.mean(dim=1)
        z_max = z.max(dim=1).values
        z_cat = torch.cat([z_mean, z_max], dim=1)
        return self.mlp(self.dropout(z_cat)).squeeze(-1)


# ───────────────────────────────────────────────────────────────
# Metric
# ───────────────────────────────────────────────────────────────
def compute_metrics(y_true, y_prob):
    y_pred = (np.array(y_prob) >= 0.5).astype(int)
    return {
        "AUROC": roc_auc_score(y_true, y_prob),
        "F1": f1_score(y_true, y_pred),
        "ACC": accuracy_score(y_true, y_pred)
    }


# ───────────────────────────────────────────────────────────────
# Main training routine
# ───────────────────────────────────────────────────────────────
def train_epicon_pr():
    print("\nTraining EpiCon-PR (Promoter fine-tuning)")

    # ─ Load data ─────────────────────────────
    matrix = np.load(DATA_PATH)
    tss_df = parse_gtf_tss(GTF_PATH)
    pos_bins = tss_df["bin"].unique()

    exclude_bins = set(pos_bins)
    for b in pos_bins:
        exclude_bins.update(range(b - 2, b + 3))  # ±2 kb 주변 제외

    candidate_neg = list(set(range(len(matrix))) - exclude_bins)
    neg_bins = sample(candidate_neg, len(pos_bins))

    all_samples = [
        (b, 1) for b in pos_bins if WINDOW_SIZE//2 <= b < len(matrix)-WINDOW_SIZE//2
    ] + [
        (b, 0) for b in neg_bins if WINDOW_SIZE//2 <= b < len(matrix)-WINDOW_SIZE//2
    ]

    train_val, test = train_test_split(all_samples, test_size=0.2, 
                                       stratify=[l for _, l in all_samples], random_state=42)
    train, val = train_test_split(train_val, test_size=0.1, 
                                  stratify=[l for _, l in train_val], random_state=42)

    train_loader = DataLoader(PromoterDataset(matrix, train), batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(PromoterDataset(matrix, val), batch_size=BATCH_SIZE)
    test_loader  = DataLoader(PromoterDataset(matrix, test), batch_size=BATCH_SIZE)

    # ─ Model ─────────────────────────────
    encoder = HMTransformerTokenEncoder().to(DEVICE)
    encoder.load_state_dict(torch.load(ENCODER_PATH, map_location=DEVICE), strict=False)
    model = EpiPRClassifier(encoder).to(DEVICE)

    optimizer = torch.optim.Adam([
        {"params": model.encoder.parameters(), "lr": 1e-5},
        {"params": model.mlp.parameters(), "lr": 1e-3}
    ])
    criterion = nn.BCEWithLogitsLoss()

    out_dir = os.path.join(SAVE_DIR, "epicon_pr")
    os.makedirs(out_dir, exist_ok=True)
    best_path = os.path.join(out_dir, "best_model.pt")

    # ─ Training Loop ─────────────────────
    best_val_auc, no_improve = 0, 0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}", ncols=100)

        for x, y in pbar:
            x, y = x.to(DEVICE), y.to(DEVICE)
            pred = model(x)
            loss = criterion(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        pbar.set_postfix(loss=f"{total_loss/len(train_loader):.4f}")

        # ─ Validation ─────────────────────
        model.eval()
        val_probs, val_labels = [], []
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(DEVICE)
                prob = torch.sigmoid(model(x)).cpu().numpy().tolist()
                val_probs.extend(prob)
                val_labels.extend(y.numpy().tolist())

        metrics = compute_metrics(val_labels, val_probs)
        print(f"  → Val AUROC: {metrics['AUROC']:.4f} | F1: {metrics['F1']:.4f} | ACC: {metrics['ACC']:.4f}")

        if metrics["AUROC"] > best_val_auc:
            best_val_auc, no_improve = metrics["AUROC"], 0
            torch.save(model.state_dict(), best_path)
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print("⏹️ Early stopping.")
                break

    # ─ Final Evaluation ───────────────────
    model.load_state_dict(torch.load(best_path, map_location=DEVICE))
    model.eval()
    test_probs, test_labels = [], []

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(DEVICE)
            prob = torch.sigmoid(model(x)).cpu().numpy().tolist()
            test_probs.extend(prob)
            test_labels.extend(y.numpy().tolist())

    test_metrics = compute_metrics(test_labels, test_probs)
    print("\nTest Metrics:", test_metrics)

    # ─ Save predictions ───────────────────
    preds = (np.array(test_probs) >= 0.5).astype(int)
    df_pred = pd.DataFrame({"true_label": test_labels, "pred_prob": test_probs, "pred_label": preds})
    pred_path = os.path.join(out_dir, "predictions.csv")
    df_pred.to_csv(pred_path, index=False)
    print(f"Saved predictions → {pred_path}")

    # ─ ROC curve ──────────────────────────
    fpr, tpr, _ = roc_curve(test_labels, test_probs)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(6,6))
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"AUROC = {roc_auc:.4f}")
    plt.plot([0,1],[0,1],color="gray",lw=1,linestyle="--")
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (Promoter Task)")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    roc_path = os.path.join(out_dir, "roc_curve.png")
    plt.savefig(roc_path)
    plt.close()
    print(f"ROC curve saved → {roc_path}")


if __name__ == "__main__":
    train_epicon_pr()