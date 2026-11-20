"""Add normalized and masked lyrics columns to a band-set dataset.

Usage (from the repo root):
    python3 src/make_variants.py --tag Death
    python3 src/make_variants.py --tag Power --top-k 20

Adds three new text columns to data/processed/dataset_<tag>_cap<cap>.parquet,
keeping every existing column, row, and split exactly as it was:
    lyrics_norm         punctuation and whitespace normalized
    lyrics_masked       proper nouns and band-specific words replaced by [MASK]
    lyrics_norm_masked  both

Then each experimental condition is a different --text-col, e.g.
    python3 src/ngram_baseline.py --tag Death --text-col lyrics_norm


Requires spaCy and its small English model:
    pip install spacy
    python3 -m spacy download en_core_web_sm
"""
import argparse
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import spacy
import spacy.lang.en.stop_words
from sklearn.feature_extraction.text import TfidfVectorizer

# Command line options
parser = argparse.ArgumentParser()
parser.add_argument("--tag", required=True, help="Death, Power, Mixed")
parser.add_argument("--cap", type=int, default=100)
parser.add_argument("--text-col", default="lyrics_clean", help="column holding cleaned lyrics")
parser.add_argument("--label-col", default="artist")
parser.add_argument("--top-k", type=int, default=20,
                    help="how many band-specific words to mask per band")
parser.add_argument("--min-songs", type=int, default=3,
                    help="a word must appear in at least this many of a band's "
                         "training songs to count as 'recurring'")
args = parser.parse_args()

data_path = Path(f"data/processed/dataset_{args.tag}_cap{args.cap}.parquet")
df = pd.read_parquet(data_path)
if args.text_col not in df.columns:
    raise SystemExit(f"No column '{args.text_col}'. Columns are: {list(df.columns)}")

MASK = "[MASK]"
texts = df[args.text_col].fillna("")