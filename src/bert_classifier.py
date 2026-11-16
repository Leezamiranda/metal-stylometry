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

