# Metal lyrics stylometry

Author attribution on metal song lyrics: a classic stylometric baseline
(function-word / character n-gram features + logistic regression or SVM, and
Burrows' Delta) compared with a fine-tuned DistilBERT classifier. Metrics:
accuracy, macro-F1, confusion matrix. To test whether the classifiers pick up
style or shared subject matter, the same pipeline is run on three band sets
(10 bands x 100 songs each): a mixed sub-genre set and two narrower
ones, death metal and power metal (see [Band sets](#band-sets)).

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
python3 -m pip install -r requirements.txt

# 1. Download the corpus and export it to Parquet (~10 s; ~500 MB file)
Rscript R/export_corpus.R               # -> data/raw/metal_lyrics.parquet

# 2. Clean + dedupe, print per-artist counts (~1.5 min, ~3 GB RAM)
python3 src/prepare_data.py             # -> data/processed/artist_counts.csv

# 3. Build the three balanced, album-split datasets (run from the repo root)
python3 src/prepare_data.py --artists "Cannibal Corpse, Deicide, Immolation, Incantation, Krisiun, Malevolent Creation, Napalm Death, Six Feet Under, Unleashed, Vader" --tag Death --cap 100
python3 src/prepare_data.py --artists "Blind Guardian, Bloodbound, Elvenking, Freedom Call, Grave Digger, HammerFall, Helloween, Iced Earth, Kamelot, Rage" --tag Power --cap 100
python3 src/prepare_data.py --artists "Accept, Korn, Iron Maiden, Venom, Sevendust, Black Label Society, King Diamond, Megadeth, W.A.S.P., GWAR" --tag Mixed --cap 100
#    -> data/processed/dataset_{Death,Power,Mixed}_cap100.parquet (+ .json with the settings)
```

The band lists and the cap are stored in `configs/band_sets.json`; the
commands above are copied from it.

Everything is deterministic (`--seed 42`), so the same commands give the same
dataset. Run `python3 src/prepare_data.py --help` for all options.

| File (data/processed/) | Contents |
|---|---|
| `songs_clean.parquet` | cleaned, deduplicated songs of all artists with >= `--min-songs` songs |
| `artist_counts.csv` | songs, albums, years, median length, genres per artist |
| `removed.parquet` | every removed song, the reason, and the song it duplicated |
| `cleaning_report.json` | row counts before/after each step |
| `dataset_<tag>_cap<C>.parquet` | final dataset; `lyrics_clean` is the model input, `split` is train/val/test |

## Band sets

Three sets of 10 bands, 100 songs per band, built with identical settings.
"Songs" is the number available after cleaning (before downsampling to 100);
"Tags" are the corpus genre labels across the band's songs, with the number
of songs carrying each label when not all of them do.

**Death** (`dataset_Death_cap100.parquet`)

| Band | Tags | Songs |
|---|---|---|
| Cannibal Corpse | Death Metal | 167 |
| Deicide | Death Metal | 126 |
| Immolation | Death Metal | 107 |
| Incantation | Death Metal | 107 |
| Krisiun | Death Metal | 110 |
| Malevolent Creation | Death Metal | 142 |
| Napalm Death | Death Metal (170), Grindcore (47) | 191 |
| Six Feet Under | Death Metal | 114 |
| Unleashed | Death Metal | 144 |
| Vader | Death Metal | 141 |

**Power** (`dataset_Power_cap100.parquet`)

| Band | Tags | Songs |
|---|---|---|
| Blind Guardian | Power Metal, Heavy Metal (17), Symphonic Metal (13), Thrash Metal (13) | 107 |
| Bloodbound | Power Metal (87), Heavy Metal (32) | 109 |
| Elvenking | Power Metal, Folk Metal (43) | 123 |
| Freedom Call | Power Metal | 119 |
| Grave Digger | Heavy Metal, Power Metal (185) | 211 |
| HammerFall | Power Metal, Heavy Metal (102) | 123 |
| Helloween | Power Metal (139), Heavy Metal (50) | 167 |
| Iced Earth | Heavy Metal, Power Metal (127) | 129 |
| Kamelot | Power Metal (115), Symphonic Metal (51), Thrash Metal (51), Progressive Metal (24), Heavy Metal (10) | 129 |
| Rage | Power Metal (269), Heavy Metal (168), Symphonic Metal (35), Thrash Metal (35), Progressive Metal (10) | 307 |

**Mixed** (`dataset_Mixed_cap100.parquet`)

| Band | Tags | Songs |
|---|---|---|
| Accept | Heavy Metal | 175 |
| Korn | Alt Metal | 162 |
| Iron Maiden | Heavy Metal | 148 |
| Venom | Heavy Metal | 143 |
| Sevendust | Alt Metal, Groove Metal (11) | 134 |
| Black Label Society | Heavy Metal (112), Southern Metal (111), Groove Metal (23) | 123 |
| King Diamond | Heavy Metal | 122 |
| Megadeth | Heavy Metal | 122 |
| W.A.S.P. | Heavy Metal | 118 |
| GWAR | Heavy Metal (96), Alt Metal (30) | 112 |

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
genre labels are joined into a `genres` column (used for picking the band sets).

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

### Experimental design: three band sets
**Research question.** Do the classifiers identify bands by style, or by
shared subject matter? A topically mixed set is compared with two topically
narrower sets. If accuracy drops sharply when all bands write about the same
things, the models were leaning on topic. Having two narrow sets acts as a
replication: one narrow set could be easy or hard for idiosyncratic reasons.

**Why death and power metal.** Metal subgenres concentrate in a few lyric
topics (Aljanaki 2024), so a subgenre is a practical stand-in for "shared
subject matter". The two are kept in separate sets because their themes differ
sharply (gore, death and anti-religion vs. fantasy, heroism and history);
merged, they would just form another mixed set.

**How bands were picked.** By hand, from the subgenre tags in a notebook,
because `prepare_data.py` selects by artist (`--artists` / `--top-n`), not by
genre. A band was excluded from the Mixed set if any of its songs carries a
Death or Power tag, so the Mixed set doesn't overlap the narrow ones.

**Why 10 x 100.** It is the largest size at which all three sets could be
filled with eligible bands (>= 3 albums, >= 100 songs after cleaning). Earlier
plans (5 bands per set, or a 6-7 band Viking set) were too small for
fine-tuning DistilBERT.

**Rejected options.**
- *Heavy Metal as a shared-topic set*: it is an umbrella label spanning many
  subgenres, so it isn't topically narrow.
- *The earlier runs* (`dataset_top20_cap157`, `dataset_viking_cap64`) are
  superseded: neither compared topic-matched sets, and the Viking set was too small.

**Known limitations of the sets.**
- The Death set is not topically uniform: Napalm Death writes political
  lyrics, Unleashed has Viking themes, Deicide is explicitly anti-religious.
- The Mixed set leans toward traditional heavy metal (6 of 10 bands are
  tagged only Heavy Metal), alongside alt/nu metal, thrash and others.

### Balancing: equal songs per artist
After cleaning, the listed artists (each must have >= 3 albums) are randomly
downsampled to the same number of songs (`--cap 100`; without `--cap`, the
smallest selected artist's count). With equal class sizes, accuracy and
macro-F1 can't be inflated by predicting the biggest artist, and the
confusion matrix rows are directly comparable. Using the same cap for all
three sets also keeps their results comparable.

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

Defaults: 1/5 of albums -> test, 1/8 of the remaining albums -> val (for
DistilBERT early stopping and hyperparameter choice). Each of the three sets
comes out at roughly 70% train / 10% val / 20% test (e.g. Death: 698 / 102 /
200 songs), and every band's test share is between about 16% and 24%. Shares
vary a little because albums differ in size.

### Evaluation plan
A single album-based split (the one above, `--seed 42`) is used during
development. Final results are averaged over 2-3 random seeds, so that a
single run doesn't decide the comparison between sets.

## Known limitations
- **Band != author.** Labels are bands. Lyricists change over a band's career,
  and some lyricists wrote for several bands (e.g. Accept and U.D.O. share
  history). Genius transcriptions are also crowd-written, so residual
  transcriber habits remain, even after normalization.
- Lines can't be recovered (see above).
- Dedup and blurb detection are heuristics; thresholds are CLI options for
  sensitivity checks (`--dup-threshold`, `--cover-threshold`, `--min-words`).
- DistilBERT's 512-token limit truncates long songs (median ~180 words is fine;
  Cradle of Filth's median is 353 words). Handle this in the model code.

