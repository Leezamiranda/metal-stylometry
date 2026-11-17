"""Fine-tuned DistilBERT classifier for band attribution.

Usage (from the repo root, on a GPU machine such as Colab):
    python src/bert_classifier.py --tag Death
    python src/bert_classifier.py --tag Power --text-col lyrics_clean --epochs 5

Uses the same datasets, splits, labels, and metrics as ngram_baseline.py,
so the two models' results can be compared directly.

"""
import argparse
import json
import math
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             classification_report, f1_score)
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          DataCollatorWithPadding, Trainer, TrainingArguments,
                          set_seed)

# ---------------------------------------------------------------------------
# Command-line options
# ---------------------------------------------------------------------------

# argparse reads the flags typed after the script name, e.g. "--tag Death".
# This lets one script run all three band sets without editing the code.
parser = argparse.ArgumentParser()
parser.add_argument("--tag", required=True, help="Death, Power, or Mixed")
parser.add_argument("--cap", type=int, default=100)
parser.add_argument("--text-col", default="lyrics_clean", help="column holding the cleaned lyrics")
parser.add_argument("--label-col", default="artist")
parser.add_argument("--model", default="distilbert-base-cased") # keeps capitalization differences
parser.add_argument("--epochs", type=int, default=5)
parser.add_argument("--lr", type=float, default=3e-5, help="learning rate")
parser.add_argument("--batch-size", type=int, default=16)
parser.add_argument("--max-len", type=int, default=512, help="max tokens per song")
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

# set seed to fix random number generator
set_seed(args.seed)

data_path = Path(f"data/processed/dataset_{args.tag}_cap{args.cap}.parquet")
# Each band set gets its own results folder so runs don't overwrite each other.
out_dir = Path(f"results/bert/{args.tag}/seed{args.seed}")
out_dir.mkdir(parents=True, exist_ok=True)

# Warn early if there is no GPU
if not torch.cuda.is_available():
    print("WARNING: no GPU found. In Colab: Runtime > Change runtime type > T4 GPU.")

# ---------------------------------------------------------------------------
# Load the data and separate the three splits
# ---------------------------------------------------------------------------

df = pd.read_parquet(data_path)
if args.text_col not in df.columns:
    raise SystemExit(f"No column '{args.text_col}'. Columns are: {list(df.columns)}")
# prepare_data.py already assigned every song to a split, grouped by album,
# so all songs from one album are in the same split
#   train: the model learns from these songs
#   val:   used to compare model settings 
#   test:  scored only once at the very end 
train = df[df["split"] == "train"]
val = df[df["split"] == "val"]
test = df[df["split"] == "test"]


# Band names turn into numbers
# sort to retain order on every run
bands = sorted(df[args.label_col].unique())
band2id = {b: i for i, b in enumerate(bands)}
id2band = {i: b for b, i in band2id.items()}

# tokenize lyrics
tokenizer = AutoTokenizer.from_pretrained(args.model)

class LyricsDataset(torch.utils.data.Dataset):
    """Holds one split's tokenized songs and band IDs in the format the
    Trainer expects: item i is a dict of token IDs plus its label."""

    def __init__(self, frame):
        # truncation=True songs are cut off at the end.
        self.enc = tokenizer(list(frame[args.text_col]), truncation=True,
                             max_length=args.max_len)
        self.labels = [band2id[b] for b in frame[args.label_col]]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        item = {k: v[i] for k, v in self.enc.items()}
        item["labels"] = self.labels[i]
        return item

train_ds = LyricsDataset(train)
val_ds = LyricsDataset(val)
test_ds = LyricsDataset(test)

# Report how many songs lost text to truncation. 
full_lengths = [len(ids) for ids in tokenizer(list(df[args.text_col]))["input_ids"]]
truncated = np.mean(np.array(full_lengths) > args.max_len)
print(f"Songs longer than {args.max_len} tokens (truncated): {truncated:.1%}")

# ---------------------------------------------------------------------------
# Load the pre-trained model with a new classification layer
# ---------------------------------------------------------------------------

model = AutoModelForSequenceClassification.from_pretrained(
    args.model, num_labels=len(bands), id2label=id2band, label2id=band2id)

def compute_metrics(eval_pred):
    """Called by the Trainer after each epoch on the validation split.
    The model outputs one score per band; the highest score is its guess."""
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {"accuracy": accuracy_score(labels, preds),
            "macro_f1": f1_score(labels, preds, average="macro")}

# ---------------------------------------------------------------------------
# Training settings
# ---------------------------------------------------------------------------
# Checkpoints go to a temporary folder that is deleted at the end

ckpt_dir = Path(f"/tmp/bert_ckpt_{args.tag}_{args.seed}")

training_args = TrainingArguments(
    output_dir=str(ckpt_dir),
    # One epoch = one pass through all training songs. Fabien et al. (2020)
    # found BERT still improving after several epochs on small datasets.
    num_train_epochs=args.epochs,
    # How big each update step is. Fine-tuning uses small values
    # so the pre-trained knowledge isn't overwritten too quickly.
    learning_rate=args.lr,
    # Songs processed together per step. 
    per_device_train_batch_size=args.batch_size,
    per_device_eval_batch_size=args.batch_size,
    # Gently shrinks weights each step, a mild guard against memorizing
    # the training songs.
    weight_decay=0.01,
    # Ramp the learning rate up over the first 10% of steps instead of
    # starting at full size, which makes early training more stable.
    # Total steps = batches per epoch x epochs.
    warmup_steps=int(0.1 * math.ceil(len(train_ds) / args.batch_size) * args.epochs),
    # Check the validation split after every epoch and save a checkpoint...
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=1,
    # reload the epoch that had the best validation
    load_best_model_at_end=True,
    metric_for_best_model="macro_f1",
    fp16=torch.cuda.is_available(),
    logging_steps=20,
    report_to="none",  # don't send logs to external tracking services
    seed=args.seed,
)

# The Trainer runs the training loop: batches, updates, evaluation, and
# checkpoints. The data collator pads each batch's songs to the length of longest song.
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    data_collator=DataCollatorWithPadding(tokenizer),
    compute_metrics=compute_metrics,
)

trainer.train()

# Keep the per-epoch validation scores to show how performance changed over training.
epoch_log = [{"epoch": r["epoch"], "val_macro_f1": round(r["eval_macro_f1"], 4),
              "val_accuracy": round(r["eval_accuracy"], 4)}
             for r in trainer.state.log_history if "eval_macro_f1" in r]
best_epoch = max(epoch_log, key=lambda r: r["val_macro_f1"])
print(f"\nBest on validation: {best_epoch}")

# ---------------------------------------------------------------------------
# Score the test data
# ---------------------------------------------------------------------------

logits = trainer.predict(test_ds).predictions
pred = [id2band[i] for i in np.argmax(logits, axis=-1)]
y_te = list(test[args.label_col])

# Same metrics as the n-gram script, so the numbers line up directly.
acc = accuracy_score(y_te, pred)
macro_f1 = f1_score(y_te, pred, average="macro")
print(f"\nTEST accuracy={acc:.3f}  macro-F1={macro_f1:.3f}  (chance = {1 / len(bands):.3f})\n")
print(classification_report(y_te, pred, zero_division=0))

# ---------------------------------------------------------------------------
# Confusion matrix
# ---------------------------------------------------------------------------
# true bands as rows, predictions as columns. 
# Comparing with the n-gram matrices side by side shows whether the models
# confuse the same pairs of bands.
fig, ax = plt.subplots(figsize=(9, 8))
ConfusionMatrixDisplay.from_predictions(y_te, pred, labels=bands,
                                        xticks_rotation=60, ax=ax, colorbar=False)
ax.set_title(f"{args.tag}: DistilBERT (test, seed {args.seed})")
fig.tight_layout()
fig.savefig(out_dir / "confusion_matrix.png", dpi=150)


# ---------------------------------------------------------------------------
# Save everything for the report > clean up checkpoints
# ---------------------------------------------------------------------------
json.dump({"tag": args.tag, "model": args.model, "seed": args.seed,
           "settings": {"epochs": args.epochs, "lr": args.lr,
                        "batch_size": args.batch_size, "max_len": args.max_len},
           "truncated_share": round(float(truncated), 4),
           "epochs": epoch_log, "best": best_epoch,
           "test_accuracy": round(acc, 4), "test_macro_f1": round(macro_f1, 4)},
          open(out_dir / "results.json", "w"), indent=2)

shutil.rmtree(ckpt_dir, ignore_errors=True)
print(f"\nSaved results and confusion matrix to {out_dir}/")