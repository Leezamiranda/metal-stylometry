"""Character n-gram baseline for band attribution.

Usage (from the repo root):
    python3 src/ngram_baseline.py --tag Death
    python3 src/ngram_baseline.py --tag Power --text-col lyrics_clean

Trains TF-IDF character n-grams + logistic regression on the train split,
picks settings on the validation split, and reports results on the test split.
"""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             classification_report, f1_score)
from sklearn.pipeline import Pipeline

# argparse reads the flags typed after the script name, e.g. "--tag Death".
# This lets one script run all three band sets without editing the code.
parser = argparse.ArgumentParser()
parser.add_argument("--tag", required=True, help="Death, Power, or Mixed")
parser.add_argument("--cap", type=int, default=100)
parser.add_argument("--text-col", default="lyrics", help="column holding the cleaned lyrics")
parser.add_argument("--label-col", default="artist")
args = parser.parse_args()

data_path = Path(f"data/processed/dataset_{args.tag}_cap{args.cap}.parquet")

# Each band set gets its own results folder so runs don't overwrite each other.
out_dir = Path(f"results/ngram/{args.tag}")
out_dir.mkdir(parents=True, exist_ok=True)

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

# X = the input text, y = the predicted band name
X_tr, y_tr = train[args.text_col], train[args.label_col]
X_val, y_tr = val[args.text_col], val[args.label_col]
X_te, y_te = test[args.text_col], test[args.label_col]

#-------------------------------------------------------------------------------------
# Model 1 : TF-IDF character n-grams → logistic regression
#--------------------------------------------------------------------------------------

def make_model(ngram_range, C):
    """Build a fresh, untrained model with the given settings.

    A Pipeline chains two steps so they always run together, in order:
    first turn text into numbers, then classify those numbers.
    """

    
    return Pipeline([
        # TfidVectorizer transforms each song into a vector
        # analyzer="char" splits text into tokens within ngram_range
        # min_df=2: ignore n-grams that appear in only one song
        # sublinear_tf=True: use log(count) instead of the raw count, so a chorus repeated 8 times doesn't count 8 times as much.
        # lowercase=False keeps capitalization as-is for use as a possible style signal
        ("tfidf", TfidfVectorizer(analyzer="char", ngram_range=ngram_range, 
                                  min_df=2, sublinear_tf=True, lowercase=False)),


        # LogisticRegression weights the n-grams for each band
        # C controls regularization - as C gets larger overfitting is more likely
        ("clf", LogisticRegression(C=C, max_iter=2000)),
    ])