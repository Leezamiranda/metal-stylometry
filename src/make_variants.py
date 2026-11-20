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