import os
import torch
import pandas as pd

# ─── Base Directory ─────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(__file__))

# ─── Data & Model Paths ─────────────────────────────────────────
DATA_PATH = os.path.join(BASE_DIR, "git", "data", "input_matrix.npy")
EIGEN_DIR = os.path.join(BASE_DIR, "git", "data", "eigenvectors_100kb")
SAVE_DIR = os.path.join(BASE_DIR, "git", "checkpoints")
ENCODER_PATH = os.path.join(SAVE_DIR, "best_model.pt")

# ─── Task-Specific Data Paths ───────────────────────────────────
# Promoter task (EpiCon-PR)
GTF_PATH = os.path.join(BASE_DIR, "git", "data", "gencode.v38.annotation.gtf")

# Chromatin state task (EpiCon-State)
CHROMHMM_GT_PATH = os.path.join(BASE_DIR, "git", "data", "bins_with_ChromHMM_GT.csv")

# ─── Training Hyperparameters ───────────────────────────────────
WINDOW_SIZE = 15
BATCH_SIZE = 256
EPOCHS = 200
LR = 1e-3
PATIENCE = 5

# ─── Filtering Thresholds ───────────────────────────────────────
NORM_THRESHOLD = 0.01
ENTROPY_THRESHOLD = 1.0

# ─── Device Setup ───────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─── GTF Parser for EpiCon-PR ───────────────────────────────────
def parse_gtf_tss(gtf_path=GTF_PATH):
    """Parse GENCODE GTF and extract TSS bin information."""
    rows = []
    with open(gtf_path, "r") as f:
        for line in f:
            if line.startswith("#"):
                continue
            fields = line.strip().split('\t')
            if len(fields) != 9 or fields[2] != "transcript":
                continue
            chrom, _, _, start, end, _, strand, _, _ = fields
            tss = int(start) if strand == "+" else int(end)
            rows.append((chrom, tss))
    df = pd.DataFrame(rows, columns=["chrom", "TSS"])
    df = df[df.chrom.str.startswith("chr") & df.chrom.str[3:].str.isnumeric()]
    df["bin"] = (df.TSS // 1000).astype(int)
    return df