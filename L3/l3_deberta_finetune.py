"""Layer 3 — DeBERTa-v1 Full Fine-Tuning for Fake Review Detection.

HPC script (A100).  Unfreezes top-N transformer layers + classifier head.
Group-aware split ensures no user leaks between train/val/test.

Usage:
    python l3_deberta_finetune.py                         # defaults
    python l3_deberta_finetune.py --unfreeze_layers 6     # unfreeze top 6
    python l3_deberta_finetune.py --batch_size 16 --grad_accum 4  # smaller GPU
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
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
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths — adjust DATA_CSV to your HPC mount
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_CSV = PROJECT_ROOT / "L1_ETL_OLAP" / "output_csv" / "reviews_enriched.csv"

OUTPUT_DIR = SCRIPT_DIR / "outputs"
PLOTS_DIR = SCRIPT_DIR / "plots"
MODEL_DIR = SCRIPT_DIR / "model"

RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data_csv", type=str, default=str(DATA_CSV))
    p.add_argument("--model_name", type=str, default="microsoft/deberta-base")
    p.add_argument("--max_length", type=int, default=256)
    p.add_argument("--unfreeze_layers", type=int, default=12,
                   help="Number of top transformer layers to unfreeze (0=head only, 12=full)")
    p.add_argument("--epochs", type=int, default=6)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--grad_accum", type=int, default=1,
                   help="Gradient accumulation steps (effective batch = batch_size * grad_accum)")
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--warmup_ratio", type=float, default=0.20)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--resume_from_checkpoint", type=str, default=None,
                   help="Path to a checkpoint dir to resume training from")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class YelpReviewDataset(Dataset):
    def __init__(self, texts: list[str], labels: list[int],
                 tokenizer, max_length: int):
        self.encodings = tokenizer(
            texts, truncation=True, padding=False, max_length=max_length,
        )
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


# ---------------------------------------------------------------------------
# Group-aware split  (same logic as L5)
# ---------------------------------------------------------------------------

def group_split(df: pd.DataFrame, seed: int = RANDOM_SEED):
    """80/10/10 stratified group split by user_id.

    Uses two rounds of StratifiedGroupKFold:
      round 1: 80% train vs 20% temp
      round 2: split temp 50/50 into val and test
    """
    df = df.reset_index(drop=True)
    labels = df["is_spam"].values
    groups = df["user_id"].values

    # Round 1: 80/20
    sgkf1 = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    train_idx, temp_idx = next(sgkf1.split(df, labels, groups))

    # Round 2: split the 20% temp into 10/10
    temp_df = df.iloc[temp_idx].reset_index(drop=True)
    temp_labels = temp_df["is_spam"].values
    temp_groups = temp_df["user_id"].values
    sgkf2 = StratifiedGroupKFold(n_splits=2, shuffle=True, random_state=seed)
    val_idx_local, test_idx_local = next(sgkf2.split(temp_df, temp_labels, temp_groups))

    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df = temp_df.iloc[val_idx_local].reset_index(drop=True)
    test_df = temp_df.iloc[test_idx_local].reset_index(drop=True)

    # Verify no user leakage
    train_users = set(train_df["user_id"])
    val_users = set(val_df["user_id"])
    test_users = set(test_df["user_id"])
    assert train_users.isdisjoint(val_users), "User leak: train ∩ val"
    assert train_users.isdisjoint(test_users), "User leak: train ∩ test"
    assert val_users.isdisjoint(test_users), "User leak: val ∩ test"

    return train_df, val_df, test_df


# ---------------------------------------------------------------------------
# Freeze / unfreeze logic
# ---------------------------------------------------------------------------

def setup_freezing(model, n_unfreeze: int, total_layers: int = 12):
    """Freeze all params, then unfreeze classifier + pooler + top-N layers."""
    for param in model.parameters():
        param.requires_grad = False

    # Always unfreeze classifier and pooler
    for name, param in model.named_parameters():
        if any(part in name for part in ["classifier", "pooler"]):
            param.requires_grad = True

    # Unfreeze top-N encoder layers (all components)
    if n_unfreeze > 0:
        unfreeze_from = total_layers - n_unfreeze
        for name, param in model.named_parameters():
            for layer_idx in range(unfreeze_from, total_layers):
                if f"encoder.layer.{layer_idx}." in name:
                    param.requires_grad = True
                    break

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


# ---------------------------------------------------------------------------
# Threshold tuning  (same approach as L5)
# ---------------------------------------------------------------------------

def find_optimal_threshold(labels: np.ndarray, probs: np.ndarray):
    """Find threshold that maximizes F1 on the precision-recall curve."""
    precision, recall, thresholds = precision_recall_curve(labels, probs)
    # precision/recall arrays are len(thresholds)+1, trim last element
    precision = precision[:-1]
    recall = recall[:-1]

    with np.errstate(divide="ignore", invalid="ignore"):
        f1_scores = 2 * precision * recall / (precision + recall)
    f1_scores = np.nan_to_num(f1_scores)

    best_idx = np.argmax(f1_scores)
    return thresholds[best_idx], f1_scores[best_idx]


# ---------------------------------------------------------------------------
# Metrics callback for Trainer
# ---------------------------------------------------------------------------

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
# Plotting
# ---------------------------------------------------------------------------

def save_plots(labels, probs, threshold_opt, plots_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_theme(style="whitegrid", palette="muted")

    # --- ROC ---
    fpr, tpr, _ = roc_curve(labels, probs)
    auc = roc_auc_score(labels, probs)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, lw=2, label=f"DeBERTa-v1 (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("L3 DeBERTa-v1 — ROC Curve")
    ax.legend(loc="lower right")
    plt.tight_layout()
    fig.savefig(plots_dir / "l3_roc.png", dpi=150)
    plt.close(fig)

    # --- PR ---
    prec, rec, _ = precision_recall_curve(labels, probs)
    ap = average_precision_score(labels, probs)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(rec, prec, lw=2, label=f"DeBERTa-v1 (AP = {ap:.3f})")
    ax.axhline(labels.mean(), color="gray", lw=1, ls="--", label="Random baseline")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("L3 DeBERTa-v1 — Precision-Recall Curve")
    ax.legend()
    plt.tight_layout()
    fig.savefig(plots_dir / "l3_pr.png", dpi=150)
    plt.close(fig)

    # --- Score distribution ---
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(probs[labels == 0], bins=50, alpha=0.6, label="Legitimate",
            color="steelblue", density=True)
    ax.hist(probs[labels == 1], bins=50, alpha=0.6, label="Spam",
            color="tomato", density=True)
    ax.axvline(0.5, color="black", lw=1.5, ls="--", label="Default threshold")
    ax.axvline(threshold_opt, color="green", lw=1.5, ls="--",
               label=f"Optimal threshold ({threshold_opt:.3f})")
    ax.set_xlabel("Predicted Spam Probability")
    ax.set_ylabel("Density")
    ax.set_title("L3 Score Distribution by True Label")
    ax.legend()
    plt.tight_layout()
    fig.savefig(plots_dir / "l3_score_distribution.png", dpi=150)
    plt.close(fig)

    # --- Confusion matrix at optimal threshold ---
    preds_opt = (probs >= threshold_opt).astype(int)
    cm = confusion_matrix(labels, preds_opt)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt=",d", cmap="Blues",
                xticklabels=["Legitimate", "Spam"],
                yticklabels=["Legitimate", "Spam"], ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix (threshold = {threshold_opt:.3f})")
    plt.tight_layout()
    fig.savefig(plots_dir / "l3_confusion_matrix.png", dpi=150)
    plt.close(fig)

    print(f"Plots saved to {plots_dir}/")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    # DeBERTa-v1 attention uses `masked_fill(~mask, finfo(query.dtype).min)`.
    # Under autocast, query is promoted to fp32 → fp32.min overflows the fp16
    # attention_scores tensor. Same issue happens with bf16. Either upgrade
    # transformers (>=4.49 has the fix) or run in fp32 (set both bf16/fp16=False).

    for d in [OUTPUT_DIR, PLOTS_DIR, MODEL_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device       : {device}")
    if device == "cuda":
        print(f"GPU          : {torch.cuda.get_device_name(0)}")
        print(f"VRAM         : {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB"
              if hasattr(torch.cuda.get_device_properties(0), "total_mem")
              else f"VRAM         : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # ── 1. Load data ──────────────────────────────────────────────────────
    print(f"\nLoading data from {args.data_csv} ...")
    df = pd.read_csv(
        args.data_csv,
        usecols=["user_id", "prod_id", "date", "rating", "review_text", "is_spam"],
        engine="python",
        on_bad_lines="skip",
    )
    df = df.dropna(subset=["review_text", "is_spam"])
    # Coerce is_spam to numeric — rows corrupted by unescaped commas in
    # review_text will have non-numeric values here; drop them.
    df["is_spam"] = pd.to_numeric(df["is_spam"], errors="coerce")
    df = df.dropna(subset=["is_spam"])
    df["is_spam"] = df["is_spam"].astype(int)
    df["review_text"] = df["review_text"].astype(str).str.strip()
    pre_filter = len(df)
    df = df[
        (df["review_text"] != "") &
        (df["review_text"].str.lower() != "nan") &
        (df["review_text"].str.len() > 10)  # drop ultra-short / garbage reviews
    ].reset_index(drop=True)
    print(f"Dropped {pre_filter - len(df):,} rows with empty/ultra-short review_text")

    print(f"Total reviews : {len(df):,}")
    print(f"Spam rate     : {df['is_spam'].mean() * 100:.1f}%")
    print(f"Unique users  : {df['user_id'].nunique():,}")

    # ── 2. Group-aware split ──────────────────────────────────────────────
    print("\nSplitting with StratifiedGroupKFold (groups=user_id) ...")
    train_df, val_df, test_df = group_split(df, seed=RANDOM_SEED)

    print(f"Train : {len(train_df):,}  (spam: {train_df['is_spam'].mean()*100:.1f}%)  "
          f"users: {train_df['user_id'].nunique():,}")
    print(f"Val   : {len(val_df):,}  (spam: {val_df['is_spam'].mean()*100:.1f}%)  "
          f"users: {val_df['user_id'].nunique():,}")
    print(f"Test  : {len(test_df):,}  (spam: {test_df['is_spam'].mean()*100:.1f}%)  "
          f"users: {test_df['user_id'].nunique():,}")

    # ── 3. Tokenizer ─────────────────────────────────────────────────────
    print(f"\nLoading tokenizer: {args.model_name} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    print("Tokenizing datasets ...")
    train_dataset = YelpReviewDataset(
        train_df["review_text"].tolist(), train_df["is_spam"].tolist(),
        tokenizer, args.max_length,
    )
    val_dataset = YelpReviewDataset(
        val_df["review_text"].tolist(), val_df["is_spam"].tolist(),
        tokenizer, args.max_length,
    )
    test_dataset = YelpReviewDataset(
        test_df["review_text"].tolist(), test_df["is_spam"].tolist(),
        tokenizer, args.max_length,
    )
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # ── 4. Model + freezing ───────────────────────────────────────────────
    print(f"\nLoading model: {args.model_name} ...")
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name, num_labels=2, ignore_mismatched_sizes=True,
    )

    trainable, total = setup_freezing(model, args.unfreeze_layers)
    print(f"Unfrozen      : classifier + pooler + top {args.unfreeze_layers} encoder layers")
    print(f"Trainable     : {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    # ── 5. Training ───────────────────────────────────────────────────────
    import transformers as _tf
    _major = int(_tf.__version__.split(".")[0])
    _eval_kw = "eval_strategy" if _major >= 5 else "evaluation_strategy"

    training_args = TrainingArguments(
        output_dir=str(MODEL_DIR / "checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        adam_epsilon=1e-6,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="cosine",
        **{_eval_kw: "epoch"},
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        bf16=False,
        fp16=False,
        logging_steps=500,
        report_to="none",
        save_total_limit=2,
        dataloader_num_workers=4,
        max_grad_norm=1.0,
        seed=RANDOM_SEED,
        dataloader_pin_memory=True,
    )

    _kwargs = dict(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        compute_metrics=make_compute_metrics(),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.patience)],
    )
    try:
        trainer = Trainer(**_kwargs, processing_class=tokenizer)
    except TypeError:
        trainer = Trainer(**_kwargs, tokenizer=tokenizer)

    print(f"\n{'='*50}")
    print(f"Starting training")
    print(f"  Epochs          : {args.epochs}")
    print(f"  Batch size      : {args.batch_size} x {args.grad_accum} accum = {args.batch_size * args.grad_accum} effective")
    print(f"  LR              : {args.lr}")
    print(f"  BF16            : {training_args.bf16}")
    print(f"  FP16            : {training_args.fp16}")
    print(f"  Unfreeze layers : {args.unfreeze_layers}")
    print(f"{'='*50}\n")

    train_result = trainer.train(
        resume_from_checkpoint=args.resume_from_checkpoint,
    )
    runtime_min = train_result.metrics["train_runtime"] / 60
    print(f"\nTraining complete! Runtime: {runtime_min:.1f} min")

    history = pd.DataFrame(trainer.state.log_history)
    history.to_csv(OUTPUT_DIR / "training_history.csv", index=False)

    # ── 6. Evaluate on test set ───────────────────────────────────────────
    print("\n" + "=" * 50)
    print("TEST SET EVALUATION")
    print("=" * 50)

    test_output = trainer.predict(test_dataset)
    logits = test_output.predictions
    probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()[:, 1]
    labels = test_df["is_spam"].values

    # Default threshold
    preds_default = (probs >= 0.5).astype(int)
    f1_default = f1_score(labels, preds_default, average="macro", zero_division=0)
    f1_spam_default = f1_score(labels, preds_default, pos_label=1, zero_division=0)

    # Optimal threshold
    threshold_opt, f1_opt = find_optimal_threshold(labels, probs)
    preds_opt = (probs >= threshold_opt).astype(int)
    f1_spam_opt = f1_score(labels, preds_opt, pos_label=1, zero_division=0)

    auc = roc_auc_score(labels, probs)
    ap = average_precision_score(labels, probs)

    print(f"AUC-ROC          : {auc:.4f}")
    print(f"Avg Precision    : {ap:.4f}")
    print(f"F1-macro @0.5    : {f1_default:.4f}")
    print(f"F1-spam  @0.5    : {f1_spam_default:.4f}")
    print(f"Optimal threshold: {threshold_opt:.4f}")
    print(f"F1-macro @opt    : {f1_opt:.4f}")
    print(f"F1-spam  @opt    : {f1_spam_opt:.4f}")
    print()
    print("Classification report @optimal threshold:")
    print(classification_report(labels, preds_opt,
                                target_names=["Legitimate", "Spam"]))

    # ── 7. Full-dataset predictions ───────────────────────────────────────
    print("Generating predictions for full dataset ...")
    full_dataset = YelpReviewDataset(
        df["review_text"].tolist(), df["is_spam"].tolist(),
        tokenizer, args.max_length,
    )
    full_output = trainer.predict(full_dataset)
    full_probs = torch.softmax(torch.tensor(full_output.predictions), dim=-1).numpy()[:, 1]

    pred_df = df[["user_id", "prod_id", "date", "is_spam"]].copy()
    pred_df["deberta_spam_prob"] = full_probs.round(6)
    pred_df["deberta_pred_default"] = (full_probs >= 0.5).astype(int)
    pred_df["deberta_pred_optimal"] = (full_probs >= threshold_opt).astype(int)

    pred_path = OUTPUT_DIR / "deberta_predictions.csv"
    pred_df.to_csv(pred_path, index=False)
    print(f"Saved {len(pred_df):,} predictions → {pred_path}")

    # ── 8. Save metrics JSON ──────────────────────────────────────────────
    metrics = {
        "model": args.model_name,
        "unfreeze_layers": args.unfreeze_layers,
        "trainable_params": trainable,
        "total_params": total,
        "max_length": args.max_length,
        "epochs_trained": int(trainer.state.epoch),
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "effective_batch_size": args.batch_size * args.grad_accum,
        "learning_rate": args.lr,
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

    metrics_path = OUTPUT_DIR / "deberta_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved → {metrics_path}")
    print(json.dumps(metrics, indent=2))

    # ── 9. Save model ────────────────────────────────────────────────────
    trainer.save_model(str(MODEL_DIR))
    tokenizer.save_pretrained(str(MODEL_DIR))
    print(f"Model saved → {MODEL_DIR}/")

    # ── 10. Threshold metadata (for L5/L6 integration) ────────────────────
    threshold_meta = {
        "model": args.model_name,
        "optimal_threshold": round(float(threshold_opt), 4),
        "f1_at_optimal": round(float(f1_opt), 4),
        "score_column": "deberta_spam_prob",
    }
    thresh_path = OUTPUT_DIR / "deberta_threshold_metadata.json"
    with open(thresh_path, "w") as f:
        json.dump(threshold_meta, f, indent=2)
    print(f"Threshold metadata saved → {thresh_path}")

    # ── 11. Plots ─────────────────────────────────────────────────────────
    save_plots(labels, probs, threshold_opt, PLOTS_DIR)

    print("\n" + "=" * 50)
    print("L3 COMPLETE")
    print("=" * 50)


if __name__ == "__main__":
    main()
