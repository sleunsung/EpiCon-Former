import os
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

from models.encoder import HMTransformerTokenEncoder
from utils.config import config
from utils.augmentations import strong_augment, TokenSupConDataset
from utils.metrics import compute_similarity_accuracy, save_attention_weights, save_embeddings, save_metrics
from utils.training import SupConLossWithFilter, EarlyStopping

def train():
    # ── Setup ─────────────────────────────────────
    matrix = np.load(config["data_path"])
    dataset = TokenSupConDataset(matrix, config["window_size"], config["entropy_threshold"])
    loader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True, num_workers=4, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HMTransformerTokenEncoder(z_dim=config["z_dim"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    criterion = SupConLossWithFilter(config["norm_threshold"])

    early_stopper = EarlyStopping(patience=config["patience"], save_path=f"{config['save_dir']}/checkpoints/best_model.pt")
    os.makedirs(config["save_dir"], exist_ok=True)

    # ── Training ───────────────────────────────────
    for epoch in range(1, config["epochs"] + 1):
        model.train()
        total_loss, total_acc, step = 0, 0, 0
        all_embeddings, all_labels, all_attn = [], [], []
        pbar = tqdm(loader, desc=f"Epoch {epoch}", ncols=100)

        for v1, v2, labels in pbar:
            v1, v2, labels = v1.to(device), v2.to(device), labels.to(device)

            z1 = model(v1, save_attn=True)
            z2 = model(v2, save_attn=False)

            embeddings, labels_filtered = criterion.filter_and_flatten(z1, z2, labels)
            if embeddings is None:
                continue

            loss = criterion(embeddings, labels_filtered)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            acc = compute_similarity_accuracy(embeddings, labels_filtered)
            total_loss += loss.item()
            total_acc += acc
            step += 1

            all_embeddings.append(embeddings.detach().cpu())
            all_labels.append(labels_filtered.detach().cpu())
            all_attn.append(model.attn_weights.copy())

            pbar.set_postfix(loss=f"{total_loss/step:.4f}", acc=f"{total_acc/step:.4f}")

        save_embeddings(epoch, all_embeddings, all_labels, config["save_dir"])
        save_attention_weights(epoch, all_attn, config["save_dir"])
        save_metrics(epoch, total_loss/step, total_acc/step, config["save_dir"])

        if early_stopper.check(total_loss / step, model):
            print("Early stopping triggered.")
            break

if __name__ == "__main__":
    train()