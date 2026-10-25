# Metal lyrics stylometry

Author attribution on metal song lyrics: a classic stylometric baseline
(function-word / character n-gram features + logistic regression or SVM, and
Burrows' Delta) compared with a fine-tuned DistilBERT classifier. Metrics:
accuracy, macro-F1, confusion matrix. A follow-up experiment narrows to
Viking-themed bands to control for topic.

Data: `corpus_metal_lyrics` from the R package
[text2map.corpora](https://gitlab.com/culturalcartography/text2map.corpora),
which holds ~193k English-language songs scraped from Genius.

## Regenerating the data

The `data/` directory is not in Git (see `.gitignore`). The lyrics are copyrighted
texts and too large to commit, so anyone cloning the repo rebuilds them:

```bash
# 0. One-time setup
#    R >= 4.5 with build tools (Ubuntu: build-essential, libcurl/ssl/xml dev libs)
Rscript -e 'install.packages(c("remotes", "nanoparquet"))'
Rscript -e 'remotes::install_gitlab("culturalcartography/text2map.corpora")'
python -m venv .venv && .venv/bin/pip install -r requirements.txt

# 1. Download the corpus and export it to Parquet (~10 s; ~500 MB file)
Rscript R/export_corpus.R               # -> data/raw/metal_lyrics.parquet

# 2. Clean + dedupe, print per-artist counts (~1.5 min, ~3 GB RAM)
.venv/bin/python src/prepare_data.py    # -> data/processed/artist_counts.csv

# 3. Build a balanced, album-split dataset
.venv/bin/python src/prepare_data.py --top-n 20
#    -> data/processed/dataset_top20_cap157.parquet (+ .json with the settings)
```

Everything is deterministic (`--seed 42`), so the same commands give the same
dataset. Run `python src/prepare_data.py --help` for all options.

| File (data/processed/) | Contents |
|---|---|
| `songs_clean.parquet` | cleaned, deduplicated songs of all artists with >= `--min-songs` songs |
| `artist_counts.csv` | songs, albums, years, median length, genres per artist |
| `removed.parquet` | every removed song, the reason, and the song it duplicated |
| `cleaning_report.json` | row counts before/after each step |
| `dataset_<tag>_cap<C>.parquet` | final dataset; `lyrics_clean` is the model input, `split` is train/val/test |

## Data decisions

Numbers are from the current corpus version. `cleaning_report.json` has the
exact counts for your run.

### Export format: Parquet, not CSV
The export is Parquet (via `nanoparquet`, a dependency-free R package). The
file is about the same size as the CSV (~500 MB, because lyrics text only
compresses ~2x), so size isn't the reason. The reasons are:
- **Types survive.** A CSV round trip turned a song actually titled "None" into
  a missing value; Parquet stores strings as strings and integers as integers.
- **Columnar reads.** `pd.read_parquet(columns=[...])` skips the large unused
  `lyrics_prepped` / `word_position` columns without parsing them.
- No quoting/escaping ambiguity for lyrics full of commas, quotes and brackets.

Cost: not human-readable or greppable, and it needs `pyarrow`. `--csv` writes a
CSV copy if you ever need one.

### 1. Duplicate rows from the genre scrape (-41,813 rows, 21.6%)
The corpus was scraped genre by genre, so an album tagged with k genres appears
k times: same URL and lyrics, different `genre`/`score`. Keeping these would
inflate artist counts (Rage 605 -> 307, Grave Digger 413 -> 211) and, worse, put
identical songs in both train and test. One row per URL is kept, and all its
genre labels are joined into a `genres` column (useful for picking Viking bands).

`song_number` is the track number within an album, not an ID; the export adds
`row_id` (original row order) for traceability.

### 2. Artist names
Names are already ASCII-folded in the corpus ("Motorhead", "Queensryche").
Normalizing case and punctuation finds 10 colliding pairs. Only pairs that are
clearly the same band are merged, via a hand-checked map (`ARTIST_ALIASES`):
Septic Flesh/Septicflesh, Rhapsody/Rhapsody of Fire (renamed in 2006),
spelling variants such as Harms Way/Harm's Way. Look-alikes that are different
bands (The Sword vs Sword, The Sorrow vs Sorrow) stay separate.

### 3. Text cleaning (raw `lyrics`, never `lyrics_prepped`)
`lyrics_prepped` is lowercased and stripped of punctuation and contractions,
which are core stylometric signals, so cleaning starts from the raw `lyrics`.
The guiding rule: **remove anything not written by the band**, and leave the
band's own text alone (case, punctuation, contractions, repeated choruses).

| Removed or normalized | Why |
|---|---|
| Genius page header: `1 Contributor` / `Translations Español ...` / `<Title> Lyrics` (148k songs) | page residue; the title would leak into features |
| Genius "About" blurbs before the first section tag, e.g. "This song is about bikers..." (~380 songs) | editorial prose by Genius users |
| All `[...]` tags (327k): `[Chorus]`, `[Verse 2: Anette Olzon]`, `[Guitar Solo: Kerry King]`, `[?]` | 36k tags name band members: a direct label leak |
| Old-style directions: `(Chorus)`, `(x2)`, `Chorus:`, `x4` | transcription markup |
| Curly quotes, `´`, dashes, `…` -> ASCII | transcriber habit, not the band's: the curly-apostrophe share ranges from 0% (Pro-Pain) to 23% (U.D.O.) |
| Cyrillic look-alike letters inside English words ("Hеll" with Cyrillic е; ~2.6k songs) | invisible artifact; a char n-gram model would learn it |
| Mojibake (`Itâ€™s` -> `It's`; ~100 songs) via `ftfy` | encoding damage, concentrated in a few artists |
| Missing space at deleted line breaks (`me?You` -> `me? You`) | scrape artifact |

**No line breaks.** The scrape replaced every line break with a space (0 of
193k songs contain `\n`). Line-level features (line length, rhymes at line
ends) are therefore unavailable. Capitalized line starts survive and act as a
weak line signal.

**No extra language filter.** `language` is "en" for all rows. A function-word
ratio filter was considered and rejected: the lowest-ratio songs are English
death/grind noun-heaps ("Dripping, melting Decaying, dying..."), which are a
writing style, not another language.

### 4. Empty, instrumental and placeholder songs (-1,912)
Songs with fewer than 30 words after cleaning are dropped (`--min-words`). This
catches instrumentals (`[Instrumental]` leaves 0 words), placeholders
(`TO BE IDENTIFIED`, `Page needs help... Paging Shell Page Bot`) and fragments.
Under ~30 words a song has only a handful of function words, too few for
frequency-based features. Genuinely short songs (grindcore) above the threshold
are kept, because brevity is part of their style.

### 5-7. Covers, re-recordings, live versions
Exact matching misses most versions (different transcriptions, extra chorus
repeats). Each song is represented as a set of word 5-grams, and two songs
match when **containment** `|A ∩ B| / min(|A|, |B|)` is high. Containment
rather than Jaccard, because a live version with extra repeats still
*contains* the original.

- **Covers** (step 5, -433): a candidate artist's song with containment >= 0.8
  against an *earlier* song by another artist, or with "cover" in the title.
  Audited examples: Primal Fear "Breaker" (Accept, 1981), Unleashed "Evil Dead"
  (Death, 1987), Wolf "Die by the Sword" (Slayer, 1983). For authorship, a
  cover carries the wrong label.
- **Same-artist versions** (step 6, -847): songs with containment >= 0.5 are
  clustered and only the earliest is kept (ties: the title without a version
  marker). Inspection of pairs showed >= 0.5 is almost always the same song
  (acoustic, re-recorded, German version); 0.3-0.5 mixes remixes with distinct
  songs that share a refrain in a concept album.
- **Version titles** (step 7, -28): "Song (Live)" / "Song - 2025 Version" when
  the artist also has "Song". Catches remixes that rearrange the lyrics too
  much for the 5-gram check.

`removed.parquet` lists every removal with its matched song, for auditing.
These checks only run for artists that could still reach `--min-songs`
(dedup only lowers counts), which keeps memory under ~3 GB.

### Balancing: equal songs per artist
After cleaning, the top-N artists (with >= 3 albums) are selected and each is
randomly downsampled to the same number of songs (default: the Nth artist's
count; top 20 -> 157 songs each). With equal class sizes, accuracy and
macro-F1 can't be inflated by predicting the biggest artist, and the
confusion matrix rows are directly comparable.

### Train/val/test split by album
Albums are kept whole: every (artist, album) pair lands in exactly one split
(`StratifiedGroupKFold`, groups = album, stratified by artist; the script
asserts this). Why not a random split:
- **Leakage.** Versions that escaped dedup, reprises, and shared refrains sit on
  the same album or across closely related ones. A random split would let the
  model match near-copies instead of learning style.
- **Album-specific vocabulary.** Concept albums share names and themes
  (Rhapsody's dragons, Sabaton's battles). A random split rewards memorizing
  album topics. An album split asks the harder, more honest question: can the
  model recognize an artist on an album it has never seen?

Defaults: 1/5 of albums -> test (~20% of each artist's songs; observed 17-22%
for the top 20), 1/8 of the remaining albums -> val (~10%, for DistilBERT
early stopping and hyperparameter choice). Shares vary a little because albums
differ in size.

For small selections (e.g. the 6-band Viking set, 64 songs each), some artists
may get no val songs. Use `--val-folds 0` and cross-validate on train
(GroupKFold by album) instead.

## Known limitations (worth stating in the report)
- **Band != author.** Labels are bands. Lyricists change over a band's career,
  and some lyricists wrote for several bands (e.g. Accept and U.D.O. share
  history). Genius transcriptions are also crowd-written, so residual
  transcriber habits remain, even after normalization.
- Lines can't be recovered (see above).
- Dedup and blurb detection are heuristics; thresholds are CLI options for
  sensitivity checks (`--dup-threshold`, `--cover-threshold`, `--min-words`).
- DistilBERT's 512-token limit truncates long songs (median ~180 words is fine;
  Cradle of Filth's median is 353 words). Handle this in the model code.

## VS Code + R notes
R is only used for downloading and exporting. The vscode-R extension's session
watcher needs the `sess` package, which ships *inside* the extension (not on
CRAN). Install it from the extension folder if the watcher fails to attach:

```bash
Rscript -e 'install.packages(c("processx", "later", "jsonlite", "rstudioapi"))'
R CMD INSTALL ~/.vscode-server/extensions/reditorsupport.r-*/dist/resources/sess
```

`languageserver` (completion/hover/diagnostics) needs libuv. If `fs` fails to
build, run `sudo apt install libuv1-dev`, or use
`USE_BUNDLED_LIBUV=1 Rscript -e 'install.packages("languageserver")'`.
