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