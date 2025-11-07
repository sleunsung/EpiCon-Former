import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from tqdm import tqdm

from utils.config import *  
from models.encoder import HMTransformerTokenEncoder  

# ───────────────────────────────────────────────────────────────
# Dataset
# ───────────────────────────────────────────────────────────────
class EpiCompDataset(Dataset):
    """Dataset for A/B compartment prediction with mean-compartment concatenation."""
    def __init__(self, matrix, bin_indices, labels, mean_comp):
        self.matrix = matrix.astype(np.float32)
        self.indices = bin_indices
        self.labels = np.array(labels).astype(np.float32)
        self.mean_comp = np.array(mean_comp).astype(np.float32)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        start = self.indices[idx] - WINDOW_SIZE // 2
        x = self.matrix[start:start + WINDOW_SIZE]
        y = self.labels[idx]
        mean_c = self.mean_comp[idx]
        return torch.from_numpy(x), torch.tensor(y), torch.tensor(mean_c)


# ───────────────────────────────────────────────────────────────
# Model
# ───────────────────────────────────────────────────────────────
class EpiCompClassifier(nn.Module):
    """Fine-tuning head for A/B compartment classification."""
    def __init__(self, encoder, hidden_dim=64):
        super().__init__()
        self.encoder = encoder
        self.mlp = nn.Sequential(
            nn.Linear(64 + 1, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x, mean_comp):
        z = self.encoder(x)
        z_pooled = z.mean(dim=1)
        concat_vec = torch.cat([z_pooled, mean_comp.unsqueeze(-1)], dim=1)
        return self.mlp(concat_vec).squeeze(-1)


# ───────────────────────────────────────────────────────────────
# Utilities
# ───────────────────────────────────────────────────────────────
def compute_metrics(y_true, y_prob):
    """Compute AUROC, F1, and Accuracy."""
    y_true = np.array(y_true)
    y_pred = (np.array(y_prob) >= 0.5).astype(int)
    return {
        "AUROC": roc_auc_score(y_true, y_prob),
        "F1": f1_score(y_true, y_pred),
        "ACC": accuracy_score(y_true, y_pred)
    }


def compute_cross_cell_mean(eigen_dir, chrom, target_cell, cell_list):
    """Compute mean eigenvector across all other cells."""
    values = []
    for cell in cell_list:
        if cell == target_cell:
            continue
        path = os.path.join(eigen_dir, cell, f"{cell}_eigen_100kb_{chrom}.csv")
        if os.path.exists(path):
            df = pd.read_csv(path, header=None).dropna()
            values.append(df[1].values)
    if not values:
        return None
    min_len = min(len(v) for v in values)
    values = [v[:min_len] for v in values]
    return np.mean(values, axis=0)


# ───────────────────────────────────────────────────────────────
# Training & Evaluation
# ───────────────────────────────────────────────────────────────
def train_epicon_comp():
    matrix = np.load(DATA_PATH)
    cell_list = ["GM12878", "HMEC", "HUVEC", "IMR90", "K562", "NHEK"]
    chroms = [f"chr{i}" for i in range(1, 23)]

    all_results = []

    for test_cell in cell_list:
        print(f"\n📌 [TEST CELL] {test_cell}")
        train_cells = [c for c in cell_list if c != test_cell]

        # ─ Train Data ─────────────────────────────
        train_idx, train_y, train_mean = [], [], []
        for cell in train_cells:
            for chrom in chroms:
                path = os.path.join(EIGEN_DIR, cell, f"{cell}_eigen_100kb_{chrom}.csv")
                if not os.path.exists(path): continue
                df = pd.read_csv(path, header=None).dropna()
                bin_idx, bin_lbl = df[0].astype(int), (df[1] > 0).astype(int)
                mean_vec = compute_cross_cell_mean(EIGEN_DIR, chrom, cell, cell_list)
                if mean_vec is None: continue
                bin_idx = bin_idx[(bin_idx >= WINDOW_SIZE // 2) & (bin_idx + WINDOW_SIZE // 2 < len(matrix))]
                min_len = min(len(bin_idx), len(bin_lbl), len(mean_vec))
                train_idx += bin_idx[:min_len].tolist()
                train_y += bin_lbl[:min_len].tolist()
                train_mean += mean_vec[:min_len].tolist()

        # ─ Test Data ─────────────────────────────
        test_idx, test_y, test_mean = [], [], []
        for chrom in chroms:
            path = os.path.join(EIGEN_DIR, test_cell, f"{test_cell}_eigen_100kb_{chrom}.csv")
            if not os.path.exists(path): continue
            df = pd.read_csv(path, header=None).dropna()
            bin_idx, bin_lbl = df[0].astype(int), (df[1] > 0).astype(int)
            mean_vec = compute_cross_cell_mean(EIGEN_DIR, chrom, test_cell, cell_list)
            if mean_vec is None: continue
            bin_idx = bin_idx[(bin_idx >= WINDOW_SIZE // 2) & (bin_idx + WINDOW_SIZE // 2 < len(matrix))]
            min_len = min(len(bin_idx), len(bin_lbl), len(mean_vec))
            test_idx += bin_idx[:min_len].tolist()
            test_y += bin_lbl[:min_len].tolist()
            test_mean += mean_vec[:min_len].tolist()

        # ─ Dataset ─────────────────────────────
        train_ds = EpiCompDataset(matrix, train_idx, train_y, train_mean)
        test_ds = EpiCompDataset(matrix, test_idx, test_y, test_mean)
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

        # ─ Model & Optimizer ─────────────────────
        encoder = HMTransformerTokenEncoder()
        encoder.load_state_dict(torch.load(ENCODER_PATH, map_location=DEVICE), strict=False)
        model = EpiCompClassifier(encoder).to(DEVICE)
        optimizer = torch.optim.Adam([
            {"params": model.encoder.parameters(), "lr": 1e-5},
            {"params": model.mlp.parameters(), "lr": 1e-4}
        ])
        criterion = nn.BCEWithLogitsLoss()

        # ─ Training Loop ─────────────────────────
        best_loss, no_improve = float("inf"), 0
        out_dir = os.path.join(SAVE_DIR, "epiconcomp_meanconcat", test_cell)
        os.makedirs(out_dir, exist_ok=True)
        best_model_path = os.path.join(out_dir, "best_model.pt")

        for epoch in range(EPOCHS):
            model.train()
            total_loss = 0
            pbar = tqdm(train_loader, desc=f"[{test_cell}] Epoch {epoch+1}/{EPOCHS}", ncols=100)

            for x, y, mean_c in pbar:
                x, y, mean_c = x.to(DEVICE), y.to(DEVICE), mean_c.to(DEVICE)
                pred = model(x, mean_c)
                loss = criterion(pred, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            pbar.set_postfix(loss=f"{total_loss/len(train_loader):.4f}")

            if total_loss < best_loss:
                best_loss, no_improve = total_loss, 0
                torch.save(model.state_dict(), best_model_path)
            else:
                no_improve += 1
                if no_improve >= PATIENCE:
                    print(f"Early stopping at epoch {epoch+1}")
                    break

        # ─ Evaluation ───────────────────────────
        model.load_state_dict(torch.load(best_model_path, map_location=DEVICE))
        model.eval()
        all_probs, all_labels = [], []

        with torch.no_grad():
            for x, y, mean_c in test_loader:
                x, y, mean_c = x.to(DEVICE), y.to(DEVICE), mean_c.to(DEVICE)
                prob = torch.sigmoid(model(x, mean_c)).cpu().numpy()
                all_probs.extend(prob.tolist())
                all_labels.extend(y.cpu().numpy().tolist())

        metrics = compute_metrics(all_labels, all_probs)
        all_results.append({"Model": "EpiConComp+Mean", "Cell": test_cell, **metrics})

    # ─ Save Results ─────────────────────────────
    df_results = pd.DataFrame(all_results)
    result_path = os.path.join(SAVE_DIR, "epiconcomp_meanconcat", "AllModel_Cellwise_summary.csv")
    df_results.to_csv(result_path, index=False)
    print(f"\n Results saved at: {result_path}")


if __name__ == "__main__":
    train_epicon_comp()