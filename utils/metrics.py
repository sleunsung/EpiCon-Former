import os
import numpy as np
import pandas as pd
import pyBigWig
from tqdm import tqdm


DATA_DIR = "/path/to/bigwig_files"
META_PATH = "/path/to/CellLine_HM_File_Matrix.csv"

# Expected CSV format:
# Each row = one cell line
# Each column = bigWig file URL for each histone modification

OUTPUT_DIR = "./input_matrix"
os.makedirs(OUTPUT_DIR, exist_ok=True)


BIN_SIZE = 1000
HISTONE_MARKS = [
    "H3K27ac", "H3K27me3", "H3K36me3",
    "H3K4me1", "H3K4me3", "H3K9me3"
]

GENOME_SIZE = {
    "chr1": 248956422, "chr2": 242193529, "chr3": 198295559, "chr4": 190214555,
    "chr5": 181538259, "chr6": 170805979, "chr7": 159345973, "chr8": 145138636,
    "chr9": 138394717, "chr10": 133797422, "chr11": 135086622, "chr12": 133275309,
    "chr13": 114364328, "chr14": 107043718, "chr15": 101991189, "chr16": 90338345,
    "chr17": 83257441, "chr18": 80373285, "chr19": 58617616, "chr20": 64444167,
    "chr21": 46709983, "chr22": 50818468
}

def extract_bigwig(args):
    cell, mark, bw_path, out_path = args
    try:
        bw = pyBigWig.open(bw_path)
    except:
        return

    rows = []
    for chrom, chrom_len in GENOME_SIZE.items():
        if chrom not in bw.chroms():
            continue

        cl = min(chrom_len, bw.chroms()[chrom])
        for start in range(0, cl, BIN_SIZE):
            end = start + BIN_SIZE
            if end > cl:
                continue
            vals = bw.values(chrom, start, end)
            m = np.nanmean(vals)
            rows.append(m if not np.isnan(m) else 0.0)

    bw.close()
    np.save(out_path, np.array(rows, dtype=np.float32))


def stage1():
    meta = pd.read_csv(META_PATH).set_index("CellLine")
    tasks = []

    for cell in meta.index:
        for mark in HISTONE_MARKS:
            url = meta.loc[cell, mark]
            if pd.isna(url):
                continue
            bw_file = url.split("/")[-1]
            bw_path = os.path.join(DATA_DIR, bw_file)
            out_path = os.path.join(OUTPUT_DIR, f"{cell}_{mark}.npy")
            tasks.append((cell, mark, bw_path, out_path))

    with ProcessPoolExecutor(max_workers=N_WORKERS) as exe:
        list(tqdm(exe.map(extract_bigwig, tasks), total=len(tasks)))


def merge_cells():
    merged = {mark: [] for mark in HISTONE_MARKS}

    for fname in os.listdir(OUTPUT_DIR):
        if not fname.endswith(".npy"):
            continue
        _, mark = fname.split("_", 1)
        mark = mark.replace(".npy", "")
        arr = np.load(os.path.join(OUTPUT_DIR, fname))
        merged[mark].append(arr)

    mark_arrays = []
    for mark in HISTONE_MARKS:
        arr = np.vstack(merged[mark])
        mark_arrays.append(arr.mean(axis=0))

    return np.vstack(mark_arrays).T


def normalize(x):
    x = np.log1p(x)
    return (x - x.mean(0)) / (x.std(0) + 1e-6)


if __name__ == "__main__":
    stage1()
    mat = merge_cells()
    mat = normalize(mat)
    np.save(os.path.join(OUTPUT_DIR, "input_matrix.npy"), mat)
