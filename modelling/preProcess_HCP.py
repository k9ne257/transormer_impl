import os
import json
import math
import argparse
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import numpy as np

try:
    import pandas as pd
except ImportError:
    pd = None

from sklearn.cluster import MiniBatchKMeans
import joblib


DEFAULT_WINDOW = 50          # W: window length (timepoints)
DEFAULT_STEP = 1             # S: step size (timepoints)
DEFAULT_K = 256              # vocab size (clusters)
DEFAULT_MAX_WINDOWS_FIT = 300_000  # cap training windows used to fit KMeans (memory/time safety)
DEFAULT_EPS = 1e-6

# token-to-char mapping: Unicode Private Use Area (safe for internal tokens)
PUA_START = 0xE000  # U+E000
PUA_END = 0xF8FF    # U+F8FF  (6400-ish chars)

SEP_SUBJECT = "|"   # subject boundary
SEP_RUN = "#"       # run boundary
START_RUN = "^"

def list_subject_files(data_dir: Path, exts=(".npy", ".parquet")) -> List[Path]:
    files = []
    for ext in exts:
        files.extend(sorted(data_dir.glob(f"*{ext}")))
    return files


def load_timeseries(path: Path) -> np.ndarray:
    """
    Load ROI time series as numpy array shape (T, R).
    Supports .npy (T,R) or (R,T) and .parquet with numeric columns.
    """
    if path.suffix == ".npy":
        arr = np.load(path)
        if arr.ndim != 2:
            raise ValueError(f"{path.name}: expected 2D array, got shape={arr.shape}")
        # Heuristic: treat larger dimension as T
        T, R = arr.shape
        if T < R:
            arr = arr.T  # convert (R,T) -> (T,R)
        return arr.astype(np.float32)

    if path.suffix == ".parquet":
        if pd is None:
            raise ImportError("pandas is required to load .parquet files. pip install pandas pyarrow")
        df = pd.read_parquet(path)
        # keep only numeric columns
        num_df = df.select_dtypes(include=[np.number])
        if num_df.shape[1] == 0:
            raise ValueError(f"{path.name}: no numeric columns found in parquet.")
        arr = num_df.to_numpy(dtype=np.float32)
        # expect (T,R)
        if arr.shape[0] < arr.shape[1]:
            # could be (R,T) if stored oddly; flip
            arr = arr.T
        return arr

    raise ValueError(f"Unsupported file type: {path}")


def zscore_ts(X: np.ndarray, eps: float = DEFAULT_EPS) -> np.ndarray:
    """Z-score per ROI (column). X shape (T,R)."""
    mu = X.mean(axis=0, keepdims=True)
    sd = X.std(axis=0, keepdims=True)
    return (X - mu) / (sd + eps)


def upper_tri_indices(R: int) -> Tuple[np.ndarray, np.ndarray]:
    """Indices for upper triangle (excluding diagonal)."""
    return np.triu_indices(R, k=1)


def dFC_vectors(X: np.ndarray, window: int, step: int, eps: float = DEFAULT_EPS) -> np.ndarray:
    """
    Compute sliding-window correlation vectors (Fisher z) from ROI time series.
    Returns V shape (num_windows, D), where D = R*(R-1)/2.
    """
    X = zscore_ts(X, eps=eps)
    T, R = X.shape
    if T < window:
        raise ValueError(f"Time series too short: T={T} < window={window}")

    iu, ju = upper_tri_indices(R)
    D = len(iu)

    num_windows = 1 + (T - window) // step
    V = np.empty((num_windows, D), dtype=np.float32)

    w = 0
    for start in range(0, T - window + 1, step):
        seg = X[start:start + window, :]  # (W,R)
        # corrcoef expects variables in rows unless rowvar=False
        C = np.corrcoef(seg, rowvar=False)  # (R,R)
        # numerical safety: clip to avoid arctanh inf
        C = np.clip(C, -0.999999, 0.999999)
        Z = np.arctanh(C).astype(np.float32)
        V[w] = Z[iu, ju]
        w += 1

    return V


def token_id_to_char(token_id: int) -> str:
    """
    Map token id -> single character.
    Uses Unicode private-use area.
    """
    codepoint = PUA_START + token_id
    if codepoint > PUA_END:
        raise ValueError(
            f"K too large for private-use mapping: token_id={token_id}, "
            f"max supported={(PUA_END - PUA_START)}"
        )
    return chr(codepoint)


def ids_to_text(ids: np.ndarray, subject_sep: str = SEP_SUBJECT, run_sep: str = SEP_RUN) -> str:
    """Convert a 1D list/array of token ids into a string of characters."""
    return "".join(token_id_to_char(int(i)) for i in ids)


def split_train_val(files: List[Path], val_ratio: float = 0.2, seed: int = 1337) -> Tuple[List[Path], List[Path]]:
    rng = np.random.default_rng(seed)
    idx = np.arange(len(files))
    rng.shuffle(idx)
    n_val = int(round(val_ratio * len(files)))
    val_idx = set(idx[:n_val].tolist())
    train_files = [f for i, f in enumerate(files) if i not in val_idx]
    val_files = [f for i, f in enumerate(files) if i in val_idx]
    return train_files, val_files


def reservoir_sample_rows(V: np.ndarray, max_rows: int, rng: np.random.Generator) -> np.ndarray:
    """Downsample rows if V is too large, without biasing too much."""
    if V.shape[0] <= max_rows:
        return V
    take = rng.choice(V.shape[0], size=max_rows, replace=False)
    return V[take]



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, required=True,
                    help="Folder containing per-subject ROI time series files (.npy or .parquet).")
    ap.add_argument("--out_dir", type=str, required=True, help="Output folder.")
    ap.add_argument("--window", type=int, default=DEFAULT_WINDOW)
    ap.add_argument("--step", type=int, default=DEFAULT_STEP)
    ap.add_argument("--k", type=int, default=DEFAULT_K)
    ap.add_argument("--val_ratio", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--max_windows_fit", type=int, default=DEFAULT_MAX_WINDOWS_FIT,
                    help="Maximum number of window-vectors used to fit KMeans.")
    ap.add_argument("--min_timepoints", type=int, default=DEFAULT_WINDOW,
                    help="Skip subjects with fewer than this many timepoints.")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = list_subject_files(data_dir)
    if not files:
        raise FileNotFoundError(f"No .npy or .parquet files found in {data_dir}")

    if args.k < 2:
        raise ValueError("k must be >= 2")
    if PUA_START + args.k > PUA_END:
        raise ValueError(f"k={args.k} too large for single-char mapping; max is {(PUA_END - PUA_START)}")

    train_files, val_files = split_train_val(files, val_ratio=args.val_ratio, seed=args.seed)
    print(f"Found {len(files)} subjects/runs total")
    print(f"Train: {len(train_files)} | Val: {len(val_files)}")
    print(f"window={args.window}, step={args.step}, K={args.k}")

    rng = np.random.default_rng(args.seed)

    # 1) Build training matrix for KMeans (sampled if too large)
    collected = []
    total_windows = 0

    for p in train_files:
        try:
            X = load_timeseries(p)
            if X.shape[0] < args.min_timepoints:
                print(f"SKIP {p.name}: T={X.shape[0]} < min_timepoints={args.min_timepoints}")
                continue
            V = dFC_vectors(X, window=args.window, step=args.step)  # (Wn, D)
            total_windows += V.shape[0]
            collected.append(V)
        except Exception as e:
            print(f"WARNING: failed {p.name}: {e}")

    if not collected:
        raise RuntimeError("No usable training subjects after filtering/errors.")

    V_all = np.vstack(collected)
    print(f"Total train windows (before sampling): {V_all.shape[0]:,} | dim={V_all.shape[1]:,}")

    V_fit = reservoir_sample_rows(V_all, max_rows=args.max_windows_fit, rng=rng)
    print(f"Windows used for KMeans fit: {V_fit.shape[0]:,}")

    # 2) Fit KMeans codebook (MiniBatch for scale)
    kmeans = MiniBatchKMeans(
        n_clusters=args.k,
        random_state=args.seed,
        batch_size=4096,
        n_init="auto",
        reassignment_ratio=0.01,
        max_no_improvement=20,
        verbose=0,
    )
    kmeans.fit(V_fit)
    joblib.dump(kmeans, out_dir / "codebook_kmeans.joblib")
    print("Saved codebook_kmeans.joblib")

    # 3) Tokenize each subject/run -> chars -> write one big text file
    def tokenize_file(p: Path) -> Optional[str]:
        X = load_timeseries(p)
        if X.shape[0] < args.min_timepoints:
            return None
        V = dFC_vectors(X, window=args.window, step=args.step)
        ids = kmeans.predict(V)  # (num_windows,)
        # Compose run string: START + tokens + RUN_SEP
        return START_RUN + ids_to_text(ids) + SEP_RUN

    train_text_parts = []
    val_text_parts = []

    train_ok = 0
    val_ok = 0

    for p in train_files:
        try:
            s = tokenize_file(p)
            if s is None:
                continue
            train_text_parts.append(s)
            train_ok += 1
        except Exception as e:
            print(f"WARNING: tokenize failed {p.name}: {e}")

    for p in val_files:
        try:
            s = tokenize_file(p)
            if s is None:
                continue
            val_text_parts.append(s)
            val_ok += 1
        except Exception as e:
            print(f"WARNING: tokenize failed {p.name}: {e}")

    # Add subject separators between runs (helps model learn boundaries)
    train_text = SEP_SUBJECT.join(train_text_parts)
    val_text = SEP_SUBJECT.join(val_text_parts)

    # Combine into one file (so your existing script can do a single read + 80/20 split)
    # If you prefer fixed split, you can also write two files and edit your training code.
    combined_text = train_text + "\n" + val_text

    out_txt = out_dir / "hcp_dfc_tokens.txt"
    out_txt.write_text(combined_text, encoding="utf-8")
    print(f"Saved tokens text: {out_txt}")
    print(f"Usable runs -> train: {train_ok}, val: {val_ok}")
    print(f"Combined text length (chars): {len(combined_text):,}")

    metadata = {
        "data_dir": str(data_dir),
        "out_dir": str(out_dir),
        "window": args.window,
        "step": args.step,
        "k": args.k,
        "val_ratio": args.val_ratio,
        "seed": args.seed,
        "min_timepoints": args.min_timepoints,
        "max_windows_fit": args.max_windows_fit,
        "separators": {
            "START_RUN": START_RUN,
            "SEP_RUN": SEP_RUN,
            "SEP_SUBJECT": SEP_SUBJECT
        },
        "token_char_mapping": {
            "unicode_private_use_start": hex(PUA_START),
            "unicode_private_use_end": hex(PUA_END),
            "max_supported_k": int(PUA_END - PUA_START)
        }
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print("Saved metadata.json")


if __name__ == "__main__":
    main()