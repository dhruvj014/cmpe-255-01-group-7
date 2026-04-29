"""Extract DeBERTa [CLS] embeddings for every review in reviews_enriched.csv.

Opt-in. Not invoked by any other layer. Run only after staging the fine-tuned
checkpoint to L3/model/checkpoint-22020/.

Outputs:
    L3/outputs/deberta_embeddings.npy           (N, 768) float32
    L3/outputs/deberta_embeddings_index.csv     (user_id, prod_id, date)

Usage:
    python3 L3/scripts/extract_embeddings.py
    python3 L3/scripts/extract_embeddings.py --batch_size 16 --max_length 192
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_CSV = PROJECT_ROOT / "L1_ETL_OLAP" / "output_csv" / "reviews_enriched.csv"
CHECKPOINT_DIR = PROJECT_ROOT / "L3" / "model"
OUTPUT_DIR = PROJECT_ROOT / "L3" / "outputs"
EMB_PATH = OUTPUT_DIR / "deberta_embeddings.npy"
INDEX_PATH = OUTPUT_DIR / "deberta_embeddings_index.csv"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, default=str(CHECKPOINT_DIR))
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--max_length", type=int, default=256)
    p.add_argument("--pooling", choices=["cls", "mean"], default="cls")
    p.add_argument("--limit", type=int, default=0, help="Process at most N rows (0=all)")
    return p.parse_args()


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_filtered() -> pd.DataFrame:
    """Same filter as l3_deberta_finetune.py — preserves alignment with predictions.csv."""
    df = pd.read_csv(
        DATA_CSV,
        usecols=["user_id", "prod_id", "date", "review_text", "is_spam"],
        engine="python",
        on_bad_lines="skip",
    )
    df = df.dropna(subset=["review_text", "is_spam"])
    df["is_spam"] = pd.to_numeric(df["is_spam"], errors="coerce")
    df = df.dropna(subset=["is_spam"])
    df["review_text"] = df["review_text"].astype(str).str.strip()
    df = df[
        (df["review_text"] != "")
        & (df["review_text"].str.lower() != "nan")
        & (df["review_text"].str.len() > 10)
    ].reset_index(drop=True)
    return df


@torch.no_grad()
def encode_batch(model, tokenizer, texts: list[str], device: str, max_length: int, pooling: str) -> np.ndarray:
    enc = tokenizer(
        texts,
        truncation=True,
        padding=True,
        max_length=max_length,
        return_tensors="pt",
    ).to(device)
    out = model(**enc)
    hidden = out.last_hidden_state  # (B, T, H)
    if pooling == "cls":
        vec = hidden[:, 0, :]
    else:
        mask = enc["attention_mask"].unsqueeze(-1).float()
        vec = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
    return vec.detach().to("cpu", dtype=torch.float32).numpy()


def main() -> None:
    args = parse_args()
    ckpt = Path(args.checkpoint)
    if not ckpt.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt}. Stage the gdrive checkpoint first."
        )

    device = get_device()
    print(f"Device: {device}")
    print(f"Loading tokenizer + model from {ckpt}")
    tokenizer = AutoTokenizer.from_pretrained(ckpt)
    model = AutoModel.from_pretrained(ckpt).to(device)
    model.eval()

    df = load_filtered()
    if args.limit > 0:
        df = df.head(args.limit).copy()
    print(f"Rows to encode: {len(df):,}  (max_length={args.max_length}, batch={args.batch_size}, pool={args.pooling})")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    n = len(df)
    hidden_dim: int | None = None
    embeddings: np.ndarray | None = None

    texts = df["review_text"].tolist()
    for start in range(0, n, args.batch_size):
        end = min(start + args.batch_size, n)
        batch = encode_batch(model, tokenizer, texts[start:end], device, args.max_length, args.pooling)
        if embeddings is None:
            hidden_dim = batch.shape[1]
            embeddings = np.zeros((n, hidden_dim), dtype=np.float32)
        embeddings[start:end] = batch
        if (start // args.batch_size) % 50 == 0:
            print(f"  {end:,}/{n:,}")

    assert embeddings is not None
    np.save(EMB_PATH, embeddings)
    df[["user_id", "prod_id", "date"]].to_csv(INDEX_PATH, index=False)
    size_mb = embeddings.nbytes / 1e6
    print(f"\nSaved embeddings: {EMB_PATH}  shape={embeddings.shape}  ~{size_mb:.0f} MB")
    print(f"Saved index:      {INDEX_PATH}")


if __name__ == "__main__":
    main()
