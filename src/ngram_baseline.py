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
parser.add_argument("--text-col", default="lyrics_clean", help="column holding the cleaned lyrics")
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
X_val, y_val = val[args.text_col], val[args.label_col]
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

#-------------------------------------------------------------------------------------
# Choose settings for n-gram lengths and C value
#--------------------------------------------------------------------------------------
search = []
for ngram_range in [(2, 4), (3, 5)]:
    for C in [1, 10, 100]:
        model = make_model(ngram_range, C).fit(X_tr, y_tr)

        # Macro-F1 averages the F1 score of each bank equally
        # If only a single band is consistently scored wrong the entire score will come down
        f1 = f1_score(y_val, model.predict(X_val), average="macro")

        search.append({"ngram_range": list(ngram_range), "C": C, "val_macro_f1": round(f1, 4)})
        print(f"ngrams={ngram_range} C={C:<4} val macro-F1={f1:.3f}")

# Keep the combination with the highest validation macro-F1.
best = max(search, key=lambda r: r["val_macro_f1"])
print(f"\nBest on validation: {best}")

#-------------------------------------------------------------------------------------
# Train with best settings and score
#--------------------------------------------------------------------------------------
final = make_model(tuple(best["ngram_range"]), best["C"])
# Add validation data to training data to maximize training data
final.fit(pd.concat([X_tr, X_val]), pd.concat([y_tr, y_val]))

# Predict a band for every song in the test data
pred = final.predict(X_te)

# Accuracy: share of test songs assigned to the correct band.
# Macro-F1: average per-band F1 
# Chance: what random guessing would score (1 / number of bands = 10%).
acc = accuracy_score(y_te, pred)
macro_f1 = f1_score(y_te, pred, average="macro")
print(f"\nTEST accuracy={acc:.3f}  macro-F1={macro_f1:.3f}  (chance = {1 / y_te.nunique():.3f})\n")

# Per-band breakdown:
#   precision: of the songs predicted as band X, how many really were X?
#   recall:    of band's songs, how many did the model find?
#   f1:        the balance of precision and recall
#   support:   how many test songs that band has
print(classification_report(y_te, pred, zero_division=0))

# ---------------------------------------------------------------------------
# Confusion matrix
# ---------------------------------------------------------------------------
# A grid with true bands as rows and predicted bands as columns. The diagonal
# holds correct predictions; any number off the diagonal shows a specific
# mix-up, e.g. "5 Vader songs were predicted as Krisiun"
fig, ax = plt.subplots(figsize=(9, 8))
ConfusionMatrixDisplay.from_predictions(y_te, pred, xticks_rotation=60, ax=ax, colorbar=False)
ax.set_title(f"{args.tag}: character n-grams (test)")
fig.tight_layout()  # adjust spacing so the rotated band names aren't cut off
fig.savefig(out_dir / "confusion_matrix.png", dpi=150)

# ---------------------------------------------------------------------------
# look for most informative n-grams per band 
# ---------------------------------------------------------------------------
# Pull the two trained steps back out of the pipeline.
vec, clf = final.named_steps["tfidf"], final.named_steps["clf"]

# The list of every n-gram the vectorizer kept, in the same order as the
# model's weights. 
names = np.array(vec.get_feature_names_out())

# clf.coef_ holds one row of weights per band. For each band:
#   np.argsort(coefs)  -> positions of the weights, from lowest to highest
#   [-15:]             -> the 15 highest-weighted positions
#   [::-1]             -> reversed, so the strongest comes first
# repr() wraps each n-gram in quotes so leading/trailing spaces are visible.
top = {band: [repr(n) for n in names[np.argsort(coefs)[-15:][::-1]]]
       for band, coefs in zip(clf.classes_, clf.coef_)}

for band, grams in top.items():
    print(f"{band:22} {' '.join(grams)}")


# ---------------------------------------------------------------------------
# Save everything for report
# ---------------------------------------------------------------------------
# One JSON file per band set, so the three runs can be compared 

json.dump({"tag": args.tag, "search": search, "best": best,
           "test_accuracy": round(acc, 4), "test_macro_f1": round(macro_f1, 4),
           "top_ngrams": top},
          open(out_dir / "results.json", "w"), indent=2)
print(f"\nSaved results and confusion matrix to {out_dir}/")
