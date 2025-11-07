import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, accuracy_score, roc_curve, auc
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
from tqdm import tqdm

from utils.config import *                # 공통 설정 (경로, 하이퍼파라미터, DEVICE)
from models.encoder import HMTransformerTokenEncoder  # encoder 구조 재사용


# ───────────────────────────────────────────────────────────────
# Dataset
# ───────────────────────────────────────────────────────────────
class StateDataset(Dataset):
    """ChromHMM state classification dataset (1kb bins)."""
    def __init__(self, matrix, indices, labels):
        self.matrix = matrix.astype(np.float32)
        self.indices = indices
        self.labels  = np.array(labels).astype(np.int64)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        center = self.indices[idx]
        start = center - WINDOW_SIZE // 2
        x = self.matrix[start:start + WINDOW_SIZE]
        y = self.labels[idx]
        return torch.from_numpy(x), torch.tensor(y)


# ───────────────────────────────────────────────────────────────
# Attention Pooling + Classifier
# ───────────────────────────────────────────────────────────────
class AttentionPooling(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.query = nn.Linear(dim, 1)

    def forward(self, x):
        score = self.query(x)
        weights = torch.softmax(score, dim=1)
        return (x * weights).sum(dim=1)


class EpiStateClassifier(nn.Module):
    """EpiCon-Former encoder + attention pooling + MLP classifier."""
    def __init__(self, encoder, num_classes):
        super().__init__()
        self.encoder = encoder
        self.pool = AttentionPooling(64)
        self.mlp = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        z = self.encoder(x)
        z_pooled = self.pool(z)
        return self.mlp(z_pooled)


# ───────────────────────────────────────────────────────────────
# Training routine
# ───────────────────────────────────────────────────────────────
def train_epicon_state():
    print("\nTraining EpiCon-State (ChromHMM fine-tuning)")

    # ─ Load data ─────────────────────────────
    matrix = np.load(DATA_PATH)
    gt_df = pd.read_csv(CHROMHMM_GT_PATH)
    gt_df = gt_df[gt_df["bin_id"] < len(matrix)].reset_index(drop=True)

    # Class map
    classes = pd.unique(gt_df["state_group"])
    class2idx = {c: i for i, c in enumerate(classes)}
    idx2class = {i: c for c, i in class2idx.items()}
    print("Classes:", class2idx)

    # Split by chromosome
    chr_train = "chr1"
    chr_valid = [f"chr{i}" for i in range(2, 22)] + ["chrX"]
    chr_test  = "chr22"

    train_idx = gt_df[gt_df["bin_chr"] == chr_train]["bin_id"].values
    train_lbl = [class2idx[x] for x in gt_df[gt_df["bin_chr"] == chr_train]["state_group"].values]

    valid_idx = gt_df[gt_df["bin_chr"].isin(chr_valid)]["bin_id"].values
    valid_lbl = [class2idx[x] for x in gt_df[gt_df["bin_chr"].isin(chr_valid)]["state_group"].values]

    test_idx  = gt_df[gt_df["bin_chr"] == chr_test]["bin_id"].values
    test_lbl  = [class2idx[x] for x in gt_df[gt_df["bin_chr"] == chr_test]["state_group"].values]

    # ─ Dataset & Loader ─────────────────────
    train_loader = DataLoader(StateDataset(matrix, train_idx, train_lbl), batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(StateDataset(matrix, valid_idx, valid_lbl), batch_size=BATCH_SIZE)
    test_loader  = DataLoader(StateDataset(matrix, test_idx,  test_lbl),  batch_size=BATCH_SIZE)

    # ─ Model ────────────────────────────────
    encoder = HMTransformerTokenEncoder().to(DEVICE)
    encoder.load_state_dict(torch.load(ENCODER_PATH, map_location=DEVICE), strict=False)
    model = EpiStateClassifier(encoder, num_classes=len(classes)).to(DEVICE)

    optimizer = torch.optim.Adam([
        {"params": model.encoder.parameters(), "lr": 1e-5},
        {"params": model.mlp.parameters(), "lr": 1e-3},
        {"params": model.pool.parameters(), "lr": 1e-3}
    ])
    criterion = nn.CrossEntropyLoss()

    out_dir = os.path.join(SAVE_DIR, "epicon_state")
    os.makedirs(out_dir, exist_ok=True)
    best_path = os.path.join(out_dir, "best_model.pt")

    # ─ Training Loop ────────────────────────
    best_val_f1, no_improve = 0, 0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}", ncols=100)

        for x, y in pbar:
            x, y = x.to(DEVICE), y.to(DEVICE)
            out = model(x)
            loss = criterion(out, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        pbar.set_postfix(loss=f"{total_loss/len(train_loader):.4f}")

        # Validation
        model.eval()
        all_probs, all_labels = [], []
        with torch.no_grad():
            for x, y in valid_loader:
                x = x.to(DEVICE)
                logits = model(x)
                prob = torch.softmax(logits, dim=1).cpu().numpy()
                all_probs.append(prob)
                all_labels.append(y.numpy())
        all_probs = np.concatenate(all_probs)
        all_labels = np.concatenate(all_labels)
        preds = all_probs.argmax(axis=1)
        val_f1 = f1_score(all_labels, preds, average="macro")
        val_acc = accuracy_score(all_labels, preds)
        print(f"  → Val F1: {val_f1:.4f} | ACC: {val_acc:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1, no_improve = val_f1, 0
            torch.save(model.state_dict(), best_path)
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print("Early stopping.")
                break

    # ─ Evaluation ───────────────────────────
    model.load_state_dict(torch.load(best_path, map_location=DEVICE))
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(DEVICE)
            logits = model(x)
            prob = torch.softmax(logits, dim=1).cpu().numpy()
            all_probs.append(prob)
            all_labels.append(y.numpy())
    all_probs = np.concatenate(all_probs)
    all_labels = np.concatenate(all_labels)
    preds = all_probs.argmax(axis=1)
    test_f1 = f1_score(all_labels, preds, average="macro")
    test_acc = accuracy_score(all_labels, preds)
    print(f"\nTest F1: {test_f1:.4f} | Test ACC: {test_acc:.4f}")

    # ─ Save predictions ─────────────────────
    df_pred = pd.DataFrame(all_probs, columns=[f"prob_{idx2class[i]}" for i in range(len(classes))])
    df_pred["true_label"] = [idx2class[i] for i in all_labels]
    df_pred["pred_label"] = [idx2class[i] for i in preds]
    pred_path = os.path.join(out_dir, "test_predictions.csv")
    df_pred.to_csv(pred_path, index=False)
    print(f"Saved predictions → {pred_path}")

    # ─ ROC Curve ────────────────────────────
    y_true_bin = label_binarize(all_labels, classes=list(range(len(classes))))
    fpr, tpr, roc_auc = {}, {}, {}
    for i in range(len(classes)):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], all_probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    plt.figure(figsize=(6,6))
    for i in range(len(classes)):
        plt.plot(fpr[i], tpr[i], lw=2, label=f"{idx2class[i]} (AUC={roc_auc[i]:.2f})")
    plt.plot([0,1],[0,1],"k--",lw=1,label="Random")
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title("ROC Curve per Chromatin State Class")
    plt.legend(loc="lower right", fontsize=8)
    plt.grid(True)
    plt.tight_layout()
    roc_path = os.path.join(out_dir, "roc_curve.png")
    plt.savefig(roc_path)
    plt.close()
    print(f"ROC curve saved → {roc_path}")


if __name__ == "__main__":
    train_epicon_state()