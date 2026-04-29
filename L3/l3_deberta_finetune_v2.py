"""Layer 3 v2 — DeBERTa Multimodal Fine-Tuning for Fake Review Detection.

Adds three things on top of l3_deberta_finetune.py, none of which require
changing the data pipeline:

  1. Focal Loss + class weighting   → fights the ~13-20% imbalance directly.
  2. Multimodal early fusion        → concat [CLS] embedding with per-review,
                                      per-user, per-product behavioural meta.
  3. Layer-wise LR decay (LLRD)     → preserves low-layer pre-training while
                                      aggressively tuning the head.

Defaults to microsoft/deberta-base (v1, bf16=False) so it runs in the same
regime that already works. Pass --model_name microsoft/deberta-v3-base
--bf16 to try v3; v3 + bf16 sidesteps the v3+fp16 NaN issue and is the safer
way to retry v3.

LEAKAGE NOTE: reviewer_profiles.{spam_count,spam_rate} and
seller_profiles.{spam_reviews,spam_rate,suspicious_reviewer_fraction} are
derived from labels across the full corpus → excluded.

Usage:
    python l3_deberta_finetune_v2.py
    python l3_deberta_finetune_v2.py --model_name microsoft/deberta-v3-base --bf16
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import Dataset
from transformers import (
    AutoModel,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

warnings.filterwarnings("ignore")

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_DATA_DIR = PROJECT_ROOT / "L1_ETL_OLAP" / "output_csv"

RANDOM_SEED = 42

# Per-review numeric features (already in reviews_enriched.csv). No labels used.
REVIEW_META = [
    "rating", "review_length", "word_count", "exclamation_count",
    "question_count", "capital_ratio", "avg_word_length",
    "day_of_week", "month",
]
# Per-user behavioral features. Excludes spam_count / spam_rate (label leak).
USER_META = [
    "review_count", "avg_rating", "rating_std", "avg_review_length",
    "avg_word_count", "unique_sellers", "tenure_days", "reviews_per_week",
    "max_seller_fraction", "avg_days_between_reviews", "burst_score",
    "rating_entropy",
]
# Per-product features. Excludes spam_reviews / spam_rate / suspicious_reviewer_fraction.
PROD_META = [
    "total_reviews", "unique_reviewers", "avg_rating", "rating_std",
    "active_days", "review_velocity",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, default=str(DEFAULT_DATA_DIR),
                   help="Folder with reviews_enriched.csv, reviewer_profiles.csv, seller_profiles.csv")
    p.add_argument("--output_root", type=str, default=str(SCRIPT_DIR),
                   help="Where outputs_v2/, plots_v2/, model_v2/ get created")
    p.add_argument("--model_name", type=str, default="microsoft/deberta-base")
    p.add_argument("--max_length", type=int, default=256)
    p.add_argument("--epochs", type=int, default=6)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--grad_accum", type=int, default=1)
    p.add_argument("--lr", type=float, default=2e-5,
                   help="Base LR for top transformer layer; LLRD decays from here.")
    p.add_argument("--head_lr", type=float, default=1e-4,
                   help="LR for classifier + meta MLP (separate from LLRD).")
    p.add_argument("--llrd_decay", type=float, default=0.9)
    p.add_argument("--warmup_ratio", type=float, default=0.10)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--focal_gamma", type=float, default=1.5)
    p.add_argument("--meta_hidden", type=int, default=64)
    p.add_argument("--resume_from_checkpoint", type=str, default=None)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Data loading + merge
# ---------------------------------------------------------------------------

def load_and_merge(data_dir: Path) -> tuple[pd.DataFrame, list[str]]:
    reviews_csv = data_dir / "reviews_enriched.csv"
    reviewer_csv = data_dir / "reviewer_profiles.csv"
    seller_csv = data_dir / "seller_profiles.csv"
    print(f"Loading {reviews_csv} ...")
    df = pd.read_csv(reviews_csv, engine="python", on_bad_lines="skip")
    df = df.dropna(subset=["review_text", "is_spam", "user_id", "prod_id"])
    df["is_spam"] = pd.to_numeric(df["is_spam"], errors="coerce")
    df = df.dropna(subset=["is_spam"])
    df["is_spam"] = df["is_spam"].astype(int)
    df["review_text"] = df["review_text"].astype(str).str.strip()
    pre = len(df)
    df = df[(df["review_text"] != "") &
            (df["review_text"].str.lower() != "nan") &
            (df["review_text"].str.len() > 10)].reset_index(drop=True)
    print(f"  Dropped {pre - len(df):,} empty/short rows; {len(df):,} remain")

    print(f"Loading {reviewer_csv} ...")
    rev = pd.read_csv(reviewer_csv)
    rev = rev[["user_id"] + USER_META]
    rev = rev.add_prefix("user_")
    rev = rev.rename(columns={"user_user_id": "user_id"})

    print(f"Loading {seller_csv} ...")
    sel = pd.read_csv(seller_csv)
    sel = sel[["prod_id"] + PROD_META]
    sel = sel.add_prefix("prod_")
    sel = sel.rename(columns={"prod_prod_id": "prod_id"})

    df = df.merge(rev, on="user_id", how="left")
    df = df.merge(sel, on="prod_id", how="left")

    meta_cols = (
        REVIEW_META
        + [f"user_{c}" for c in USER_META]
        + [f"prod_{c}" for c in PROD_META]
    )
    # Coerce + impute median-fill done later using train-only stats.
    for c in meta_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    print(f"Total reviews : {len(df):,}")
    print(f"Spam rate     : {df['is_spam'].mean() * 100:.1f}%")
    print(f"Meta features : {len(meta_cols)}")
    return df, meta_cols


# ---------------------------------------------------------------------------
# Group-aware split (unchanged from v1)
# ---------------------------------------------------------------------------

def group_split(df: pd.DataFrame, seed: int = RANDOM_SEED):
    df = df.reset_index(drop=True)
    labels = df["is_spam"].values
    groups = df["user_id"].values
    sgkf1 = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    train_idx, temp_idx = next(sgkf1.split(df, labels, groups))
    temp_df = df.iloc[temp_idx].reset_index(drop=True)
    sgkf2 = StratifiedGroupKFold(n_splits=2, shuffle=True, random_state=seed)
    val_idx, test_idx = next(
        sgkf2.split(temp_df, temp_df["is_spam"].values, temp_df["user_id"].values)
    )
    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df = temp_df.iloc[val_idx].reset_index(drop=True)
    test_df = temp_df.iloc[test_idx].reset_index(drop=True)

    assert set(train_df["user_id"]).isdisjoint(val_df["user_id"])
    assert set(train_df["user_id"]).isdisjoint(test_df["user_id"])
    assert set(val_df["user_id"]).isdisjoint(test_df["user_id"])
    return train_df, val_df, test_df


# ---------------------------------------------------------------------------
# Dataset with text + meta tensor
# ---------------------------------------------------------------------------

class FusedDataset(Dataset):
    def __init__(self, texts, labels, meta, tokenizer, max_length):
        self.enc = tokenizer(
            texts, truncation=True, padding=False, max_length=max_length,
        )
        self.labels = labels
        self.meta = meta.astype(np.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.enc.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        item["meta_features"] = torch.tensor(self.meta[idx], dtype=torch.float32)
        return item


class FusedCollator:
    """Pad text fields dynamically; stack meta_features."""
    def __init__(self, tokenizer):
        self.pad_id = tokenizer.pad_token_id

    def __call__(self, features):
        labels = torch.stack([f["labels"] for f in features])
        meta = torch.stack([f["meta_features"] for f in features])
        max_len = max(len(f["input_ids"]) for f in features)
        input_ids = torch.full((len(features), max_len), self.pad_id, dtype=torch.long)
        attn = torch.zeros((len(features), max_len), dtype=torch.long)
        for i, f in enumerate(features):
            n = len(f["input_ids"])
            input_ids[i, :n] = f["input_ids"]
            attn[i, :n] = f["attention_mask"]
        return {
            "input_ids": input_ids,
            "attention_mask": attn,
            "meta_features": meta,
            "labels": labels,
        }


# ---------------------------------------------------------------------------
# Model: DeBERTa text + MLP meta → fused classifier head
# ---------------------------------------------------------------------------

class FocalLoss(nn.Module):
    def __init__(self, alpha: torch.Tensor, gamma: float = 1.5):
        super().__init__()
        self.register_buffer("alpha", alpha)
        self.gamma = gamma

    def forward(self, logits, target):
        logp = F.log_softmax(logits, dim=-1)
        p = logp.exp()
        target_logp = logp.gather(1, target.unsqueeze(1)).squeeze(1)
        target_p = p.gather(1, target.unsqueeze(1)).squeeze(1)
        alpha_t = self.alpha[target]
        loss = -alpha_t * (1 - target_p) ** self.gamma * target_logp
        return loss.mean()


class DebertaFusedClassifier(nn.Module):
    def __init__(self, model_name: str, num_meta: int, meta_hidden: int,
                 class_weights: torch.Tensor, focal_gamma: float):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name, dtype=torch.float32)
        h = self.backbone.config.hidden_size
        self.meta_mlp = nn.Sequential(
            nn.Linear(num_meta, meta_hidden),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(meta_hidden, meta_hidden),
            nn.GELU(),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(h + meta_hidden, h),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(h, 2),
        )
        self.loss_fn = FocalLoss(alpha=class_weights, gamma=focal_gamma)

    def forward(self, input_ids, attention_mask, meta_features, labels=None):
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0]
        m = self.meta_mlp(meta_features)
        logits = self.classifier(torch.cat([cls, m], dim=-1))
        loss = None
        if labels is not None:
            loss = self.loss_fn(logits, labels)
        return {"loss": loss, "logits": logits}


# ---------------------------------------------------------------------------
# LLRD optimizer construction
# ---------------------------------------------------------------------------

def build_llrd_optimizer(model: DebertaFusedClassifier, base_lr: float,
                         head_lr: float, decay: float, weight_decay: float):
    """Layer-wise LR decay across backbone; flat head_lr for meta_mlp + classifier."""
    no_decay = ("bias", "LayerNorm.weight", "LayerNorm.bias")
    backbone = model.backbone
    n_layers = backbone.config.num_hidden_layers

    groups = []

    # Embeddings: lowest LR.
    emb_lr = base_lr * (decay ** (n_layers + 1))
    emb_params = list(backbone.embeddings.named_parameters())
    groups.append({"params": [p for n, p in emb_params if not any(nd in n for nd in no_decay)],
                   "lr": emb_lr, "weight_decay": weight_decay})
    groups.append({"params": [p for n, p in emb_params if any(nd in n for nd in no_decay)],
                   "lr": emb_lr, "weight_decay": 0.0})

    # Encoder layers: index 0 = lowest, n-1 = highest. Top layer gets base_lr.
    for i, layer in enumerate(backbone.encoder.layer):
        layer_lr = base_lr * (decay ** (n_layers - 1 - i))
        params = list(layer.named_parameters())
        groups.append({"params": [p for n, p in params if not any(nd in n for nd in no_decay)],
                       "lr": layer_lr, "weight_decay": weight_decay})
        groups.append({"params": [p for n, p in params if any(nd in n for nd in no_decay)],
                       "lr": layer_lr, "weight_decay": 0.0})

    # Heads (meta_mlp + classifier): flat, higher LR.
    head_params = list(model.meta_mlp.named_parameters()) + list(model.classifier.named_parameters())
    groups.append({"params": [p for n, p in head_params if not any(nd in n for nd in no_decay)],
                   "lr": head_lr, "weight_decay": weight_decay})
    groups.append({"params": [p for n, p in head_params if any(nd in n for nd in no_decay)],
                   "lr": head_lr, "weight_decay": 0.0})

    return torch.optim.AdamW(groups, eps=1e-6)


# ---------------------------------------------------------------------------
# Threshold + metrics
# ---------------------------------------------------------------------------

def find_optimal_threshold(labels, probs):
    precision, recall, thresholds = precision_recall_curve(labels, probs)
    precision, recall = precision[:-1], recall[:-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        f1s = 2 * precision * recall / (precision + recall)
    f1s = np.nan_to_num(f1s)
    i = np.argmax(f1s)
    return thresholds[i], f1s[i]


def make_compute_metrics():
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()[:, 1]
        preds = (probs >= 0.5).astype(int)
        try:
            auc = roc_auc_score(labels, probs)
            ap = average_precision_score(labels, probs)
        except ValueError:
            auc = ap = 0.0
        return {
            "f1_macro": f1_score(labels, preds, average="macro", zero_division=0),
            "f1_spam": f1_score(labels, preds, pos_label=1, zero_division=0),
            "auc_roc": auc,
            "avg_precision": ap,
        }
    return compute_metrics


# ---------------------------------------------------------------------------
# Plots (mirrors v1)
# ---------------------------------------------------------------------------

def save_plots(labels, probs, threshold_opt, plots_dir, tag="DeBERTa-v2"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_theme(style="whitegrid", palette="muted")

    fpr, tpr, _ = roc_curve(labels, probs)
    auc = roc_auc_score(labels, probs)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, lw=2, label=f"{tag} (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR"); ax.set_title("L3 v2 — ROC")
    ax.legend(loc="lower right"); plt.tight_layout()
    fig.savefig(plots_dir / "l3v2_roc.png", dpi=150); plt.close(fig)

    prec, rec, _ = precision_recall_curve(labels, probs)
    ap = average_precision_score(labels, probs)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(rec, prec, lw=2, label=f"{tag} (AP = {ap:.3f})")
    ax.axhline(labels.mean(), color="gray", lw=1, ls="--", label="Random")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision"); ax.set_title("L3 v2 — PR")
    ax.legend(); plt.tight_layout()
    fig.savefig(plots_dir / "l3v2_pr.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(probs[labels == 0], bins=50, alpha=0.6, label="Legit",
            color="steelblue", density=True)
    ax.hist(probs[labels == 1], bins=50, alpha=0.6, label="Spam",
            color="tomato", density=True)
    ax.axvline(0.5, color="black", lw=1.5, ls="--", label="0.5")
    ax.axvline(threshold_opt, color="green", lw=1.5, ls="--",
               label=f"Optimal ({threshold_opt:.3f})")
    ax.set_xlabel("P(spam)"); ax.set_ylabel("Density")
    ax.set_title("L3 v2 — Score Distribution"); ax.legend()
    plt.tight_layout()
    fig.savefig(plots_dir / "l3v2_score_distribution.png", dpi=150); plt.close(fig)

    preds_opt = (probs >= threshold_opt).astype(int)
    cm = confusion_matrix(labels, preds_opt)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt=",d", cmap="Blues",
                xticklabels=["Legit", "Spam"], yticklabels=["Legit", "Spam"], ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title(f"Confusion @ {threshold_opt:.3f}")
    plt.tight_layout()
    fig.savefig(plots_dir / "l3v2_confusion_matrix.png", dpi=150); plt.close(fig)
    print(f"Plots saved to {plots_dir}/")


# ---------------------------------------------------------------------------
# Custom Trainer that uses our pre-built optimizer
# ---------------------------------------------------------------------------

class FusedTrainer(Trainer):
    def __init__(self, *args, llrd_config=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._llrd_config = llrd_config or {}

    def create_optimizer(self):
        if self.optimizer is None:
            self.optimizer = build_llrd_optimizer(self.model, **self._llrd_config)
        return self.optimizer


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    output_root = Path(args.output_root)
    OUTPUT_DIR = output_root / "outputs_v2"
    PLOTS_DIR = output_root / "plots_v2"
    MODEL_DIR = output_root / "model_v2"
    for d in [OUTPUT_DIR, PLOTS_DIR, MODEL_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device       : {device}")
    if device == "cuda":
        props = torch.cuda.get_device_properties(0)
        print(f"GPU          : {torch.cuda.get_device_name(0)}")
        print(f"VRAM         : {props.total_memory / 1e9:.1f} GB")

    df, meta_cols = load_and_merge(Path(args.data_dir))

    train_df, val_df, test_df = group_split(df, RANDOM_SEED)
    print(f"Train : {len(train_df):,} ({train_df['is_spam'].mean()*100:.1f}% spam)")
    print(f"Val   : {len(val_df):,} ({val_df['is_spam'].mean()*100:.1f}% spam)")
    print(f"Test  : {len(test_df):,} ({test_df['is_spam'].mean()*100:.1f}% spam)")

    # Train-only normalization (median impute → robust z-score).
    train_meta_raw = train_df[meta_cols].astype(np.float32)
    median = train_meta_raw.median()
    mad = (train_meta_raw - median).abs().median().replace(0, 1.0)

    def normalize(d: pd.DataFrame) -> np.ndarray:
        x = d[meta_cols].astype(np.float32).fillna(median)
        x = (x - median) / (1.4826 * mad)
        return x.clip(-5, 5).to_numpy()

    train_meta = normalize(train_df)
    val_meta = normalize(val_df)
    test_meta = normalize(test_df)

    print(f"\nTokenizer: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    train_ds = FusedDataset(train_df["review_text"].tolist(),
                            train_df["is_spam"].tolist(),
                            train_meta, tokenizer, args.max_length)
    val_ds = FusedDataset(val_df["review_text"].tolist(),
                          val_df["is_spam"].tolist(),
                          val_meta, tokenizer, args.max_length)
    test_ds = FusedDataset(test_df["review_text"].tolist(),
                           test_df["is_spam"].tolist(),
                           test_meta, tokenizer, args.max_length)
    collator = FusedCollator(tokenizer)

    # Class weights: inverse-frequency, normalized to mean 1.0.
    pos_rate = train_df["is_spam"].mean()
    w0 = 1.0 / (1.0 - pos_rate)
    w1 = 1.0 / pos_rate
    s = (w0 + w1) / 2
    class_weights = torch.tensor([w0 / s, w1 / s], dtype=torch.float32)
    print(f"Class weights: legit={class_weights[0]:.3f}  spam={class_weights[1]:.3f}")
    print(f"Focal gamma  : {args.focal_gamma}")

    print(f"\nBuilding model: {args.model_name} + {len(meta_cols)}-dim meta MLP")
    model = DebertaFusedClassifier(
        args.model_name, num_meta=len(meta_cols),
        meta_hidden=args.meta_hidden,
        class_weights=class_weights, focal_gamma=args.focal_gamma,
    )
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"Params trainable: {n_trainable:,} / {n_total:,}")

    llrd_config = dict(
        base_lr=args.lr, head_lr=args.head_lr,
        decay=args.llrd_decay, weight_decay=args.weight_decay,
    )
    print(f"LLRD: base_lr={args.lr}  head_lr={args.head_lr}  decay={args.llrd_decay}")

    import transformers as _tf
    _eval_kw = "eval_strategy" if int(_tf.__version__.split(".")[0]) >= 5 else "evaluation_strategy"

    training_args = TrainingArguments(
        output_dir=str(MODEL_DIR / "checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,  # used only by scheduler; optimizer is custom
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="cosine",
        **{_eval_kw: "epoch"},
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        bf16=args.bf16,
        fp16=args.fp16,
        logging_steps=500,
        report_to="none",
        save_total_limit=2,
        dataloader_num_workers=4,
        max_grad_norm=1.0,
        seed=RANDOM_SEED,
        dataloader_pin_memory=True,
    )

    trainer = FusedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        compute_metrics=make_compute_metrics(),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.patience)],
        llrd_config=llrd_config,
    )

    print(f"\n{'='*50}\nTraining\n{'='*50}")
    train_result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    runtime_min = train_result.metrics["train_runtime"] / 60
    print(f"Runtime: {runtime_min:.1f} min")

    pd.DataFrame(trainer.state.log_history).to_csv(
        OUTPUT_DIR / "training_history.csv", index=False
    )

    # ── Test ──
    print(f"\n{'='*50}\nTEST\n{'='*50}")
    test_out = trainer.predict(test_ds)
    probs = torch.softmax(torch.tensor(test_out.predictions), dim=-1).numpy()[:, 1]
    labels = test_df["is_spam"].values

    preds_default = (probs >= 0.5).astype(int)
    f1_default = f1_score(labels, preds_default, average="macro", zero_division=0)
    f1_spam_default = f1_score(labels, preds_default, pos_label=1, zero_division=0)
    threshold_opt, f1_opt = find_optimal_threshold(labels, probs)
    preds_opt = (probs >= threshold_opt).astype(int)
    f1_spam_opt = f1_score(labels, preds_opt, pos_label=1, zero_division=0)
    auc = roc_auc_score(labels, probs)
    ap = average_precision_score(labels, probs)

    print(f"AUC-ROC          : {auc:.4f}")
    print(f"AP               : {ap:.4f}")
    print(f"F1-macro @0.5    : {f1_default:.4f}")
    print(f"F1-spam  @0.5    : {f1_spam_default:.4f}")
    print(f"Optimal threshold: {threshold_opt:.4f}")
    print(f"F1-macro @opt    : {f1_opt:.4f}")
    print(f"F1-spam  @opt    : {f1_spam_opt:.4f}")
    print()
    print(classification_report(labels, preds_opt, target_names=["Legit", "Spam"]))

    # ── Full-corpus predictions ──
    full_meta = normalize(df)
    full_ds = FusedDataset(df["review_text"].tolist(), df["is_spam"].tolist(),
                           full_meta, tokenizer, args.max_length)
    full_out = trainer.predict(full_ds)
    full_probs = torch.softmax(torch.tensor(full_out.predictions), dim=-1).numpy()[:, 1]
    pred_df = df[["user_id", "prod_id", "date", "is_spam"]].copy()
    pred_df["deberta_spam_prob"] = full_probs.round(6)
    pred_df["deberta_pred_default"] = (full_probs >= 0.5).astype(int)
    pred_df["deberta_pred_optimal"] = (full_probs >= threshold_opt).astype(int)
    pred_path = OUTPUT_DIR / "deberta_v2_predictions.csv"
    pred_df.to_csv(pred_path, index=False)
    print(f"Predictions → {pred_path}")

    metrics = {
        "model": args.model_name,
        "fusion": True,
        "num_meta_features": len(meta_cols),
        "meta_features": meta_cols,
        "focal_gamma": args.focal_gamma,
        "class_weights": class_weights.tolist(),
        "llrd_decay": args.llrd_decay,
        "base_lr": args.lr,
        "head_lr": args.head_lr,
        "max_length": args.max_length,
        "epochs_trained": int(trainer.state.epoch),
        "batch_size": args.batch_size,
        "effective_batch_size": args.batch_size * args.grad_accum,
        "bf16": args.bf16,
        "fp16": args.fp16,
        "train_size": len(train_df),
        "val_size": len(val_df),
        "test_size": len(test_df),
        "auc_roc": round(auc, 4),
        "avg_precision": round(ap, 4),
        "f1_macro_default": round(f1_default, 4),
        "f1_spam_default": round(f1_spam_default, 4),
        "optimal_threshold": round(float(threshold_opt), 4),
        "f1_macro_optimal": round(float(f1_opt), 4),
        "f1_spam_optimal": round(f1_spam_opt, 4),
        "runtime_minutes": round(runtime_min, 1),
    }
    with open(OUTPUT_DIR / "deberta_v2_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps({k: v for k, v in metrics.items()
                      if k not in ("meta_features",)}, indent=2))

    trainer.save_model(str(MODEL_DIR))
    tokenizer.save_pretrained(str(MODEL_DIR))
    # Save normalization stats so inference can reproduce features.
    norm_stats = {
        "meta_cols": meta_cols,
        "median": median.to_dict(),
        "mad": mad.to_dict(),
    }
    with open(MODEL_DIR / "meta_norm_stats.json", "w") as f:
        json.dump(norm_stats, f, indent=2)
    print(f"Model + norm stats → {MODEL_DIR}/")

    with open(OUTPUT_DIR / "deberta_v2_threshold_metadata.json", "w") as f:
        json.dump({
            "model": args.model_name,
            "fusion": True,
            "optimal_threshold": round(float(threshold_opt), 4),
            "f1_at_optimal": round(float(f1_opt), 4),
            "score_column": "deberta_spam_prob",
        }, f, indent=2)

    save_plots(labels, probs, threshold_opt, PLOTS_DIR,
               tag=f"DeBERTa+meta ({Path(args.model_name).name})")

    print(f"\n{'='*50}\nL3 v2 COMPLETE\n{'='*50}")


if __name__ == "__main__":
    main()
