# Metal lyrics stylometry

This project asks whether text classifiers can identify a metal band from its
lyrics by writing style, or only by subject matter. A character n-gram model
and a fine-tuned DistilBERT model are trained on three sets of 10 bands, with
the lyrics in four versions that remove topic words, transcription-dependent
punctuation, or both. The n-gram model outperforms DistilBERT in every
condition. Masking topic words lowers DistilBERT's score on the topically
mixed set by about 0.17 macro-F1 but barely affects the n-gram model, and all
scores stay far above chance (0.10) after both changes.

## Research question and motivation

Authorship attribution assumes that a classifier learns how an author writes.
Metal is a hard test of that assumption: subgenres concentrate on a few lyric
topics, so a classifier can succeed by recognizing what a band
writes about (gore, Satan, dragons, battles) instead of how it writes. The
project compares a topically mixed set of bands with two topically narrower
sets, death metal and power metal. If accuracy holds up in the narrow sets
and after topic words are masked, the models are using more than topic.
Having two narrow sets acts as a replication.

A second question was added after instructor feedback. The lyrics are
crowd-transcribed on Genius, so punctuation may reflect the person who
transcribed a song rather than the band. Normalizing punctuation tests how
much of the signal depends on it.

## Data

### Source

The data is the Metal Lyrics Corpus (`corpus_metal_lyrics`) from the R package
[text2map.corpora](https://gitlab.com/culturalcartography/text2map.corpora):
193,623 rows of English-language song lyrics scraped from Genius, with artist,
album, year, and genre labels. The lyrics are copyrighted, so neither the
corpus nor the derived datasets are included in this repository; the
`data/` folder is gitignored and is rebuilt with the commands under
[Reproducing the results](#reproducing-the-results).

### Cleaning

`src/prepare_data.py` cleans the corpus in the following steps (counts are
from `data/processed/cleaning_report.json`):

| Step | Rows removed |
|---|---|
| 1. Duplicate rows: the corpus was scraped genre by genre, so an album tagged with k genres appears k times. One row per lyrics URL is kept, with all its genre labels joined. | 41,813 (21.6%) |
| 4. Songs with fewer than 30 words after cleaning (instrumentals, placeholders, fragments) | 1,912 |
| 5. Covers: songs that largely contain an earlier song by another artist, or have "cover" in the title | 433 |
| 6. Near-duplicates by the same artist (re-recordings, live and acoustic versions); only the earliest version is kept | 847 |
| 7. Alternate versions identified by title, e.g. "Song (Live)" when "Song" is also present | 28 |

Steps 5-7 compare songs as sets of word 5-grams and use containment,
`|A ∩ B| / min(|A|, |B|)`, rather than Jaccard similarity, because a live
version with extra chorus repeats still contains the original. They run only
for artists with enough songs to be candidates (at least 50).

Text cleaning (steps 2-3) starts from the raw `lyrics` column, not the
corpus's `lyrics_prepped`, which is lowercased and stripped of punctuation and
contractions. The rule is to remove text the band did not write and leave the
band's own text unchanged (case, punctuation, contractions, repeated choruses):

- Genius page headers (contributor counts, translation lists, "<Title>
  Lyrics"), which would leak the song title into the features.
- Genius "About" blurbs written by site users.
- All bracketed section tags such as `[Chorus]` or `[Verse 2: <name>]`. Many
  name band members, which would leak the label directly.
- Old-style directions such as `(Chorus)` or `(x2)`.
- Cyrillic letters that look identical to Latin ones inside English words,
  and encoding damage (mojibake, fixed with `ftfy`).
- Curly quotes and apostrophes, dashes, and ellipses, converted to ASCII.
  Their use depends on the transcriber, not the band.

Artist names that differ only in spelling or punctuation were merged when
they are clearly the same band (for example Septic Flesh and Septicflesh), and
kept separate otherwise.

The original scrape replaced every line break with a space, so line-level
features such as line length or end rhyme are not available.

### Band sets

Three sets of 10 bands were chosen by hand from the corpus genre tags. The
death and power metal sets are kept separate because their themes differ
sharply. A band was excluded from the Mixed set if any of its songs carries a
Death or Power tag. Heavy Metal was not used as a narrow set because it is an
umbrella label covering many subgenres. The band lists are stored in
`configs/band_sets.json`.

| Death | Power | Mixed |
|---|---|---|
| Cannibal Corpse | Blind Guardian | Accept |
| Deicide | Bloodbound | Korn |
| Immolation | Elvenking | Iron Maiden |
| Incantation | Freedom Call | Venom |
| Krisiun | Grave Digger | Sevendust |
| Malevolent Creation | HammerFall | Black Label Society |
| Napalm Death | Helloween | King Diamond |
| Six Feet Under | Iced Earth | Megadeth |
| Unleashed | Kamelot | W.A.S.P. |
| Vader | Rage | GWAR |

Each band is randomly downsampled to 100 songs, so every set has 1,000 songs
and equal class sizes. Ten bands with 100 songs each was the largest size at
which all three sets could be filled; earlier plans with 5 bands per set, or
a 6-7 band Viking-themed set, were too small for fine-tuning DistilBERT.

### Train/validation/test split

Songs are split by album: every album lands entirely in one split
(`StratifiedGroupKFold`, grouped by album, stratified by band). A random split
would place re-recordings, reprises, and songs from the same concept album in
both train and test, letting a model match near-copies or album-specific
vocabulary instead of learning style. The album split asks whether the model
recognizes a band on an album it has never seen.

One fifth of each band's albums goes to test and one eighth of the remaining
albums to validation. Each set comes out at about 70% train, 10% validation,
and 20% test (for example, Death: 698 / 102 / 200 songs), and every band's
test share is between 16% and 24%.

## Method

### Text versions

`src/make_variants.py` adds three text versions to each dataset, so every
model is trained on four versions of the same songs:

| Version | Column | Change |
|---|---|---|
| Original | `lyrics_clean` | cleaned lyrics as described above |
| Normalized | `lyrics_norm` | every punctuation mark replaced by a space, except apostrophes inside words; whitespace collapsed; capitalization kept |
| Masked | `lyrics_masked` | topic words replaced by `[MASK]` |
| Normalized + masked | `lyrics_norm_masked` | both changes |

Masking replaces two kinds of words:

- **Proper nouns**, as tagged by spaCy (`en_core_web_sm`). Tagging runs on the
  original text, because punctuation and context help the tagger.
- **Each band's 20 most distinctive recurring words.** All of a band's
  training songs are joined into one document and scored with TF-IDF against
  the other bands in the set. The highest-scoring words that occur in at least
  3 of the band's training songs are kept, excluding spaCy stop words and
  words containing apostrophes. Only the training split is used, so the test
  set does not influence what is masked.

Masking replaces about 9-10% of the words in each set. The word lists are
saved in `results/masking/`.

### Models

- **Character n-grams + logistic regression** (`src/ngram_baseline.py`):
  TF-IDF over character n-grams with case preserved, sublinear term frequency,
  and n-grams that occur in at least 2 songs. The n-gram range (2-4, 3-5, 4-6)
  and regularization strength C (1 to 100) are chosen by validation macro-F1;
  the final model is retrained on train + validation and scored on test.
- **DistilBERT** (`src/bert_classifier.py`): `distilbert-base-cased` (cased,
  so capitalization is visible to the model), fine-tuned for 5 epochs with
  learning rate 3e-5, batch size 16, and a maximum of 512 tokens per song
  (0.8-5.3% of the original songs per set are truncated). The epoch with the best
  validation macro-F1 is kept. It was run on a GPU in Google Colab with seed
  42.

### Evaluation

The metric is macro-F1 on the test split, which weights every band equally.
With 10 balanced classes, chance is 0.10. Each run also saves accuracy, a
confusion matrix, and per-song predictions.

## Results

Test macro-F1 by model, band set, and text version (from
`results/*/*/*/results.json`; DistilBERT is a single run with seed 42):

| Model | Set | Original | Normalized | Masked | Normalized + masked |
|---|---|---|---|---|---|
| Char n-grams | Mixed | 0.737 | 0.692 | 0.722 | 0.650 |
| Char n-grams | Death | 0.802 | 0.820 | 0.792 | 0.790 |
| Char n-grams | Power | 0.780 | 0.744 | 0.750 | 0.724 |
| DistilBERT | Mixed | 0.598 | 0.525 | 0.426 | 0.522 |
| DistilBERT | Death | 0.627 | 0.671 | 0.623 | 0.610 |
| DistilBERT | Power | 0.609 | 0.552 | 0.553 | 0.524 |

Key findings:

- **The n-gram model outperforms DistilBERT in all 12 conditions**, by 0.13 to
  0.30 macro-F1.
- **Masking affects DistilBERT more than the n-gram model on the Mixed set.**
  DistilBERT's Mixed score falls from 0.598 to 0.426 (-0.17), while the n-gram
  model's falls from 0.737 to 0.722 (-0.02). In the narrower sets, masking
  changes both models by at most 0.06.
- **Normalization has small average effects** (-0.02 for the n-gram model and
  -0.03 for DistilBERT, averaged over the three sets) but large effects for
  individual bands: the n-gram model's F1 for Accept drops from 0.76 to 0.46
  (-0.31).
- **Every score stays far above chance** after both changes; the lowest is
  0.522, against a chance level of 0.10.

`notebooks/analysis.ipynb` builds this table, the change from the original
lyrics, and per-band F1 for every condition from the files in `results/`.

## Limitations

- **Single seed for DistilBERT.** Each condition was run once, so small
  differences may be noise. For example, DistilBERT's Mixed score is lower
  with masking alone (0.426) than with masking and normalization (0.522).
- **No line breaks.** The original scrape removed them, so line-level
  features are unavailable.
- **Topic is only partly controlled.** Bands within a subgenre still differ in
  theme: in the Death set, Napalm Death writes political lyrics, Unleashed
  uses Viking themes, and Deicide is explicitly anti-religious. The Mixed set
  leans toward traditional heavy metal (6 of 10 bands are tagged only Heavy
  Metal).
- **The masking word lists include some general words** (for example "oh",
  "yeah", "like"), so masking removes some non-topical words as well.
- **Band is not author.** Labels are bands, while lyricists change over a
  band's career, and Genius transcriptions add the habits of individual
  transcribers.

## Repository structure

```
configs/
  band_sets.json        band lists and songs-per-band cap for the three sets
R/
  export_corpus.R       downloads the corpus and exports it to Parquet
src/
  prepare_data.py       cleaning, deduplication, balancing, album split
  make_variants.py      normalized and masked text versions
  ngram_baseline.py     character n-gram + logistic regression model
  bert_classifier.py    fine-tuned DistilBERT model (GPU)
notebooks/
  metal_stylometry_BERT_classifier.ipynb   Colab notebook used for the DistilBERT runs
  analysis.ipynb                           results tables and per-band F1
results/
  ngram/<set>/<text version>/          results.json, predictions.csv, confusion_matrix.png
  bert/<set>/<text version>/seed42/    same files for DistilBERT
  masking/<set>_masked_words.json      masked words per band
requirements.txt
```

`data/` (raw corpus and processed datasets) is created by the pipeline and is
not tracked.

## Reproducing the results

All commands run from the repository root. Every step uses a fixed seed (42),
so the same commands produce the same datasets.

### Environment

```bash
# R >= 4.5 with build tools (Ubuntu: build-essential and the libcurl, libssl, libxml2 dev packages)
Rscript -e 'install.packages(c("remotes", "nanoparquet"))'
Rscript -e 'remotes::install_gitlab("culturalcartography/text2map.corpora")'

# Python (versions in requirements.txt were tested with Python 3.14)
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 -m spacy download en_core_web_sm
```

### 1. Export the corpus

```bash
Rscript R/export_corpus.R      # -> data/raw/metal_lyrics.parquet (~500 MB)
```

### 2. Build the three datasets

```bash
python3 src/prepare_data.py --artists "Cannibal Corpse, Deicide, Immolation, Incantation, Krisiun, Malevolent Creation, Napalm Death, Six Feet Under, Unleashed, Vader" --tag Death --cap 100
python3 src/prepare_data.py --artists "Blind Guardian, Bloodbound, Elvenking, Freedom Call, Grave Digger, HammerFall, Helloween, Iced Earth, Kamelot, Rage" --tag Power --cap 100
python3 src/prepare_data.py --artists "Accept, Korn, Iron Maiden, Venom, Sevendust, Black Label Society, King Diamond, Megadeth, W.A.S.P., GWAR" --tag Mixed --cap 100
```

Each run writes `data/processed/dataset_<set>_cap100.parquet` with a settings
file alongside it, plus the cleaning outputs (`songs_clean.parquet`,
`artist_counts.csv`, `removed.parquet`, `cleaning_report.json`). Cleaning
takes about 1.5 minutes and 3 GB of RAM per run.

### 3. Add the text versions

```bash
python3 src/make_variants.py --tag Death
python3 src/make_variants.py --tag Power
python3 src/make_variants.py --tag Mixed
```

### 4. Character n-gram model

```bash
for set in Death Power Mixed; do
  for col in lyrics_clean lyrics_norm lyrics_masked lyrics_norm_masked; do
    python3 src/ngram_baseline.py --tag $set --text-col $col
  done
done
```

### 5. DistilBERT (GPU)

The DistilBERT runs used Google Colab with a T4 GPU, through
`notebooks/metal_stylometry_BERT_classifier.ipynb`. Because `data/` is not in
the repository, the local `data/` folder (containing `processed/` with the
four-version datasets from step 3) was uploaded to Google Drive. The notebook
then:

1. mounts Google Drive, clones this repository, and links the Drive folder
   into the clone as `data/` (`ln -s <Drive folder> data`; change the path to
   the folder's location in your own Drive);
2. installs `transformers` and `accelerate`;
3. runs `src/bert_classifier.py` once per band set and text version, which is
   equivalent to:

   ```bash
   for set in Death Power Mixed; do
     for col in lyrics_clean lyrics_norm lyrics_masked lyrics_norm_masked; do
       python3 src/bert_classifier.py --tag $set --text-col $col
     done
   done
   ```

4. copies `results/` back to Google Drive, since files in the Colab session
   are deleted when it ends.

Results are written to `results/bert/<set>/<text version>/seed42/`; copy them
into `results/` in the local repository before running the analysis.

### 6. Analysis

Open `notebooks/analysis.ipynb` and run all cells. It reads only the files in
`results/`.

## Use of AI tools

Claude Code was used to set up the R environment and to write the corpus
export (`R/export_corpus.R`) and the data-cleaning pipeline
(`src/prepare_data.py`). The author specified the requirements, asked it to
inspect the data and explain each decision, reviewed every finding, and made
the design decisions.
