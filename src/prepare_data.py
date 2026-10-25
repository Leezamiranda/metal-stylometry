"""Clean the metal lyrics corpus and build an author-attribution dataset.

Two-step usage (from the project root):

    # 1. Clean + dedupe, then report per-artist song counts
    python src/prepare_data.py

    # 2. Pick N from the report, cap every artist at the same size, split by album
    python src/prepare_data.py --top-n 20            # cap = Nth artist's count
    python src/prepare_data.py --top-n 20 --cap 120
    python src/prepare_data.py --min-songs 40 --tag viking \
        --artists "Amon Amarth,Enslaved,Grand Magus,Eluveitie,Bathory,Tyr"

Input:  data/raw/metal_lyrics.parquet (written by R/export_corpus.R)
Output: data/processed/
    songs_clean.parquet        cleaned, deduplicated songs of candidate artists
    artist_counts.csv          song/album counts per candidate artist
    cleaning_report.json       rows removed at each step
    removed.parquet            every removed song, the reason, and what it duplicated
    dataset_top{N}_cap{C}.parquet   final dataset with a `split` column

See README.md ("Data decisions") for the reasoning behind each step.
"""

import argparse
import json
import re
from pathlib import Path

import ftfy
import numpy as np
import pandas as pd
from scipy.sparse.csgraph import connected_components
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "metal_lyrics.parquet"
OUT = ROOT / "data" / "processed"

# Same band under two names (checked by hand: same albums/era/members).
# Look-alikes that are *different* bands are deliberately not merged:
# The Sword / Sword, The Sorrow / Sorrow, The Prophecy / Prophecy, Nine / nine,
# and Luca Turilli's Rhapsody (a 2011 spin-off) stays separate.
ARTIST_ALIASES = {
    "Rhapsody": "Rhapsody of Fire",  # renamed in 2006; surfaced as a "cover"
    "Septic Flesh": "Septicflesh",
    "AEnimus": "Aenimus",
    "A Life Divided": "A Life [Divided]",
    "Arka'n": "Arkan",
    "Harms Way": "Harm's Way",
    "The Mad Capsule Markets": "The Mad Capsule Market's",
}

# --- text cleaning ---------------------------------------------------------

# Genius page header: "[N Contributor(s)] [Translations <langs>] <Title> Lyrics"
HEADER = re.compile(r"^(?P<prefix>.{0,250}?) ?Lyrics(?=[\s\[\"“(]|$)")
MOJIBAKE = re.compile(r"â€|Ã[\x80-\xbf¡-ÿ]|Â[\x80-\xbf ]")
# Partial mojibake that ftfy can't repair because a byte was already lost.
MOJIBAKE_LEFTOVER = {'â€"': "-", "â€™": "'", "â€œ": '"', "â€¦": "...", "â€": ""}
# Cyrillic letters that look identical to Latin ones ("Hеll" with Cyrillic е).
HOMOGLYPHS = str.maketrans("аеорсухіјѕАВЕКМНОРСТХІЈЅ", "aeopcyxijsABEKMHOPCTXIJS")
CYRILLIC = re.compile(r"[Ѐ-ӿ]")
TYPOGRAPHY = str.maketrans({
    "’": "'", "‘": "'", "‛": "'", "′": "'", "`": "'", "´": "'",
    "“": '"', "”": '"', "„": '"', "″": '"',
    "–": "-", "—": "-", "−": "-",
    " ": " ", " ": " ", " ": " ",
    "​": None, "‌": None, "‍": None, "﻿": None,
})
BRACKET_TAG = re.compile(r"\[[^\[\]]*\]")
STRUCTURAL_TAG = re.compile(
    r"\[(?:verse|chorus|pre|post|intro|outro|bridge|hook|refrain|part|interlude|"
    r"breakdown|instrumental|solo|guitar|spoken)", re.I)
# Genius "About" text that sometimes precedes the lyrics.
BLURB = re.compile(
    r"\b(?:this|the) (?:song|track|album|single|record|lyrics?)\b"
    r"|\b(?:was|were) (?:released|written|recorded|inspired)\b"
    r"|\b(?:stands for|refers to|is about|are about|written by)\b", re.I)
# Old-style performance directions written in parentheses or as labels.
PAREN_DIRECTION = re.compile(
    r"\(\s*(?:x\s?\d+|\d+\s?x|repeat[^)]*|(?:pre-?)?chorus[^)]*|verse[^)]*|bridge[^)]*|"
    r"(?:guitar )?solo[^)]*|instrumental[^)]*|intro|outro|refrain[^)]*)\s*\)", re.I)
LABEL_DIRECTION = re.compile(
    r"(?<![\w'])(?:Pre-?Chorus|Chorus|Verse(?: \d+)?|Bridge|Refrain|Intro|Outro|Hook)"
    r"\s*\d*\s*:")
REPEAT_COUNT = re.compile(r"(?<![\w'])(?:[x×]\s?\d+|\d+\s?[x×])(?![\w'])")


def strip_header(text, title):
    """Remove the Genius page header, but only if it plausibly is one."""
    m = HEADER.match(text)
    if not m:
        return text, False
    prefix = m.group("prefix").lower()
    first_word = re.sub(r"[^\w]", "", title.split()[0].lower()) if title.split() else ""
    # Genius's displayed title can differ from song_title ("LI(E) FE" vs
    # "LI (E) FE"), so also accept a prefix about as short as the title.
    plausible = ("contributor" in prefix or "translations" in prefix
                 or (first_word and first_word in re.sub(r"[^\w ]", "", prefix))
                 or len(prefix) <= len(title) + 20)
    if not plausible:
        return text, False
    return text[m.end():], True


def strip_blurb(text):
    """Drop editorial prose before the first section tag (Genius 'About' text)."""
    m = STRUCTURAL_TAG.search(text)
    if not m or m.start() == 0:
        return text, False
    lead = text[: m.start()]
    if BLURB.search(lead):
        return text[m.start():], True
    return text, False


def clean_lyrics(raw, title):
    """Return (clean_text, flags). Keeps case, punctuation and contractions."""
    flags = {}
    text = raw
    flags["fixed_mojibake"] = bool(MOJIBAKE.search(text))
    if flags["fixed_mojibake"]:
        text = ftfy.fix_text(text, uncurl_quotes=False)
        for bad, good in MOJIBAKE_LEFTOVER.items():
            text = text.replace(bad, good)
    # Stray Cyrillic look-alikes in a Latin-script song; genuinely Cyrillic
    # lyrics (>5% of letters) are left alone.
    n_cyr = len(CYRILLIC.findall(text))
    flags["fixed_homoglyphs"] = 0 < n_cyr < 0.05 * sum(ch.isalpha() for ch in text)
    if flags["fixed_homoglyphs"]:
        text = text.translate(HOMOGLYPHS)
    # Transcriber conventions, not the band's: normalize to ASCII forms.
    text = text.translate(TYPOGRAPHY).replace("…", "...")
    text, flags["had_header"] = strip_header(text.strip(), title or "")
    text, flags["had_blurb"] = strip_blurb(text.strip())
    flags["n_tags"] = len(BRACKET_TAG.findall(text))
    text = BRACKET_TAG.sub(" ", text)
    text = text.replace("[", " ").replace("]", " ")  # unbalanced leftovers
    text = PAREN_DIRECTION.sub(" ", text)
    text = LABEL_DIRECTION.sub(" ", text)
    text = REPEAT_COUNT.sub(" ", text)
    # Line breaks were deleted in the scrape; restore the missing space where a
    # line ended in punctuation ("me?You" -> "me? You", "blood,I'll" -> ...).
    text = re.sub(r"([?!,;:])(?=[A-Za-z])", r"\1 ", text)
    text = re.sub(r"(?<=[a-z]{2})\.(?=[A-Z][a-z])", ". ", text)
    text = re.sub(r"(?<=\w)\(", " (", text)
    text = re.sub(r"\)(?=\w)", ") ", text)
    text = re.sub(r"\(\s*\)", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text, flags


# --- deduplication ---------------------------------------------------------

VERSION_MARKER = re.compile(
    r"[\(\[\-–—].*\b(?:live|demo|remaster(?:ed)?|re-?record(?:ed|ing)?|acoustic|unplugged|"
    r"version|edit|mix|remix|instrumental|orchestral|rehearsal|session|bootleg|"
    r"alternate|radio|extended|\d{4})\b", re.I)
COVER_MARKER = re.compile(r"\bcover\b", re.I)


def base_title(title):
    t = title.lower()
    t = re.sub(r"\s*[\(\[][^\)\]]*[\)\]]", "", t)   # (Live), [Bonus Track], (Ft. X)
    t = re.sub(r"\s+[-–—]\s+.*$", "", t)           # " - 2015 Remaster"
    return re.sub(r"[^a-z0-9]+", " ", t).strip()


def shingle_matrix(texts, k):
    """Binary song x word-k-gram matrix (hashed), on lowercased letters only."""
    norm = (texts.str.lower().str.replace("'", "", regex=False)
            .str.replace(r"[^a-z0-9]+", " ", regex=True))
    hv = HashingVectorizer(analyzer="word", ngram_range=(k, k), n_features=2**24,
                           token_pattern=r"\S+", binary=True, norm=None,
                           alternate_sign=False, dtype=np.float32)
    X = hv.transform(norm).tocsr()
    X.data[:] = 1.0
    return X


def containment(X_a, X_b, size_a, size_b):
    """|A ∩ B| / min(|A|, |B|) for all row pairs, as a sparse COO matrix.

    Containment rather than Jaccard because a live version with extra chorus
    repeats, or a truncated transcription, still contains the original.
    """
    S = (X_a @ X_b.T).tocoo()
    denom = np.minimum(size_a[S.row], size_b[S.col])
    S.data = S.data / np.maximum(denom, 1)
    return S


def keep_order(df):
    """Preference within a duplicate cluster: earliest year, then a title
    without a version marker, then the original corpus order."""
    return df.assign(_v=df.song_title.str.contains(VERSION_MARKER)).sort_values(
        ["year", "_v", "row_id"]).index


def dedupe_within_artist(df, X, size, threshold):
    """Cluster each artist's songs whose shingle containment >= threshold
    (union-find via connected components) and keep one song per cluster.
    Returns {dropped index: kept index}."""
    drop = {}
    pos = pd.Series(np.arange(len(df)), index=df.index)
    for _, g in df.groupby("artist"):
        if len(g) < 2:
            continue
        ix = pos[g.index].to_numpy()
        S = containment(X[ix], X[ix], size[ix], size[ix])
        mask = (S.data >= threshold) & (S.row != S.col) & (
            np.minimum(size[ix][S.row], size[ix][S.col]) >= 5)
        S.data, S.row, S.col = S.data[mask], S.row[mask], S.col[mask]
        _, labels = connected_components(S, directed=False)
        for lab in np.unique(labels):
            members = g.iloc[labels == lab]
            if len(members) > 1:
                kept, *rest = keep_order(members)
                drop.update(dict.fromkeys(rest, kept))
    return drop


def dedupe_versions_by_title(df):
    """Drop version-marked titles ("Song (Live)", "Song - Remix") when the same
    artist also has the base song. Catches remixes that rearrange lyrics too
    much for the shingle check."""
    d = df.assign(_bt=df.song_title.map(base_title),
                  _v=df.song_title.str.contains(VERSION_MARKER))
    drop = {}
    for _, g in d[d._bt != ""].groupby(["artist", "_bt"]):
        if len(g) > 1 and g._v.any():
            # Only marked titles are dropped: two unmarked "Intro"s are
            # different songs. Prefer an unmarked original over any version.
            kept = g.sort_values(["_v", "year", "row_id"]).index[0]
            drop.update({i: kept for i in g.index[g._v] if i != kept})
    return drop


def find_covers(df, cand_mask, X, size, threshold):
    """Songs by candidate artists whose lyrics match a song by another artist.

    The earlier release is treated as the original. Same-year matches are
    ambiguous (collaborations, split releases), so the candidate copy is dropped.
    Returns {dropped index: matching index, or -1 for a "cover" title}.
    """
    drop = dict.fromkeys(df.index[cand_mask & df.song_title.str.contains(COVER_MARKER)], -1)
    cand_pos = np.flatnonzero(cand_mask.to_numpy())
    artists = df.artist.to_numpy()
    years = df.year.to_numpy()
    for start in range(0, len(cand_pos), 2000):
        rows = cand_pos[start:start + 2000]
        S = containment(X[rows], X, size[rows], size)
        a, b = rows[S.row], S.col
        hit = ((S.data >= threshold) & (artists[a] != artists[b])
               & (np.minimum(size[a], size[b]) >= 5) & (years[b] <= years[a]))
        for i, j in zip(df.index[a[hit]], df.index[b[hit]]):
            drop.setdefault(i, j)
    return drop


# --- dataset construction --------------------------------------------------

def album_split(df, test_folds, val_folds, seed):
    """Assign train/val/test so every (artist, album) lands in exactly one split,
    while each artist keeps roughly the same share in each split."""
    groups = df.artist + " || " + df.album
    split = pd.Series("train", index=df.index)
    sgkf = StratifiedGroupKFold(n_splits=test_folds, shuffle=True, random_state=seed)
    _, test_ix = next(sgkf.split(df, df.artist, groups))
    split.iloc[test_ix] = "test"
    if val_folds:
        rest = df[split == "train"]
        sgkf = StratifiedGroupKFold(n_splits=val_folds, shuffle=True, random_state=seed)
        _, val_ix = next(sgkf.split(rest, rest.artist, groups[rest.index]))
        split[rest.index[val_ix]] = "val"
    return split


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--top-n", type=int, help="number of artists in the final dataset")
    ap.add_argument("--artists", help='explicit comma-separated artist list instead of '
                    '--top-n, e.g. "Amon Amarth,Enslaved,Bathory"')
    ap.add_argument("--tag", help="name for the output file (default: topN / Kartists)")
    ap.add_argument("--cap", type=int, help="songs per artist (default: smallest selected artist)")
    ap.add_argument("--min-words", type=int, default=30,
                    help="drop songs with fewer words after cleaning (default 30)")
    ap.add_argument("--min-songs", type=int, default=50,
                    help="only artists with at least this many songs are candidates")
    ap.add_argument("--min-albums", type=int, default=3,
                    help="artists need this many albums to appear in every split")
    ap.add_argument("--dup-threshold", type=float, default=0.5,
                    help="5-gram containment at which two songs are versions (default 0.5)")
    ap.add_argument("--cover-threshold", type=float, default=0.8)
    ap.add_argument("--test-folds", type=int, default=5, help="1/k of albums go to test")
    ap.add_argument("--val-folds", type=int, default=8,
                    help="1/k of remaining albums go to val; 0 disables val")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    report = {}
    removed = []

    def record(frame, idx, reason, matched=None):
        """Keep an audit trail of removed songs (and what they duplicated)."""
        r = frame.loc[idx, ["row_id", "artist", "album", "year", "song_title"]].copy()
        r["reason"] = reason
        if matched is not None:
            m = pd.Series(matched)
            ok = m >= 0
            r["matched_row_id"] = pd.NA
            r.loc[m.index[ok], "matched_row_id"] = frame.loc[m[ok].to_numpy(), "row_id"].to_numpy()
            r.loc[m.index[ok], "matched"] = (
                frame.loc[m[ok].to_numpy(), "artist"] + " | "
                + frame.loc[m[ok].to_numpy(), "song_title"] + " | "
                + frame.loc[m[ok].to_numpy(), "year"].astype(str)).to_numpy()
        removed.append(r)

    def log(step, before, after):
        report[step] = {"before": int(before), "after": int(after), "removed": int(before - after)}
        print(f"{step:<42} {before:>8,} -> {after:>8,}  (-{before - after:,})")

    cols = ["row_id", "artist", "album", "year", "song_title", "song_lyrics_url",
            "lyrics", "genre", "subgenre"]
    raw = pd.read_parquet(RAW, columns=cols)

    # 1. The corpus was scraped per genre: an album tagged with k genres appears
    #    k times (same URL and lyrics). Keep one row; collect all genre labels.
    genres = (raw.groupby("song_lyrics_url").genre
              .agg(lambda g: "; ".join(sorted(set(g)))).rename("genres"))
    df = raw.drop_duplicates("song_lyrics_url").drop(columns="genre")
    df = df.join(genres, on="song_lyrics_url").reset_index(drop=True)
    log("1. duplicate URLs (genre-scrape copies)", len(raw), len(df))
    del raw

    # 2. Artist spelling variants.
    df["artist"] = df.artist.replace(ARTIST_ALIASES)

    # 3. Clean raw lyrics (NOT lyrics_prepped: it drops case and punctuation).
    cleaned = [clean_lyrics(l, t) for l, t in zip(df.lyrics, df.song_title)]
    df["text"] = [c[0] for c in cleaned]
    df = pd.concat([df, pd.DataFrame([c[1] for c in cleaned], index=df.index)], axis=1)
    df["n_words"] = df.text.str.split().str.len()
    report["cleaning_flags"] = {k: int(df[k].sum()) for k in
                                ["fixed_mojibake", "fixed_homoglyphs", "had_header",
                                 "had_blurb"]}
    report["cleaning_flags"]["tags_removed"] = int(df.n_tags.sum())

    # 4. Empty, instrumental, placeholder ("TO BE IDENTIFIED") and fragment texts.
    n = len(df)
    record(df, df.index[df.n_words < args.min_words], "too_short")
    df = df[df.n_words >= args.min_words].reset_index(drop=True)
    log(f"4. fewer than {args.min_words} words", n, len(df))

    # Duplicates can only lower an artist's count, so the expensive checks only
    # need to run for artists that could still reach --min-songs.
    counts = df.artist.value_counts()
    cand = df.artist.isin(counts.index[counts >= args.min_songs])
    print(f"   candidate artists (>= {args.min_songs} songs before dedup): "
          f"{int((counts >= args.min_songs).sum())}")

    X = shingle_matrix(df.text, k=5)
    size = np.asarray(X.sum(axis=1)).ravel()

    # 5. Covers (lyrics written by someone else) among candidate artists.
    n = cand.sum()
    covers = find_covers(df, cand, X, size, args.cover_threshold)
    record(df, list(covers), "cover", covers)
    keep = ~df.index.isin(list(covers))
    log("5. covers / cross-artist duplicates", n, (cand & keep).sum())
    df, X, size, cand = df[keep], X[keep], size[keep], cand[keep]
    df = df[cand].reset_index(drop=True)
    X, size = X[cand.to_numpy()], size[cand.to_numpy()]

    # 6. Same song recorded more than once (re-recordings, live, demo, remix).
    n = len(df)
    drop = dedupe_within_artist(df, X, size, args.dup_threshold)
    record(df, list(drop), "near_duplicate", drop)
    keep = ~df.index.isin(list(drop))
    df, X, size = df[keep].reset_index(drop=True), X[keep], size[keep]
    log("6. near-duplicate lyrics, same artist", n, len(df))
    n = len(df)
    drop = dedupe_versions_by_title(df)
    record(df, list(drop), "version_title", drop)
    df = df.drop(list(drop)).reset_index(drop=True)
    log("7. version-marked titles of kept songs", n, len(df))

    df = df.drop(columns="lyrics").rename(columns={"text": "lyrics_clean"})
    df.to_parquet(OUT / "songs_clean.parquet", index=False)
    pd.concat(removed).to_parquet(OUT / "removed.parquet", index=False)

    stats = (df.groupby("artist")
             .agg(songs=("row_id", "size"), albums=("album", "nunique"),
                  first_year=("year", "min"), last_year=("year", "max"),
                  median_words=("n_words", "median"), genres=("genres", "first"))
             .sort_values(["songs", "albums"], ascending=False))
    stats = stats[stats.songs >= args.min_songs]
    stats.to_csv(OUT / "artist_counts.csv")
    print(f"\nArtists with >= {args.min_songs} songs after cleaning: {len(stats)}")
    print(stats.head(40).to_string())

    if args.top_n or args.artists:
        eligible = stats[stats.albums >= args.min_albums]
        if args.artists:
            names = [a.strip() for a in args.artists.split(",") if a.strip()]
            missing = [a for a in names if a not in eligible.index]
            if missing:
                raise SystemExit(f"not eligible (< {args.min_songs} songs or < "
                                 f"{args.min_albums} albums, or misspelled): {missing}")
            top = eligible.loc[names]
            tag = args.tag or f"{len(names)}artists"
        else:
            top = eligible.head(args.top_n)
            if len(top) < args.top_n:
                raise SystemExit(f"only {len(top)} artists have >= {args.min_albums} albums")
            tag = args.tag or f"top{args.top_n}"
        cap = args.cap or int(top.songs.min())
        if cap > top.songs.min():
            raise SystemExit(f"--cap {cap} exceeds {top.songs.idxmin()}'s {top.songs.min()} songs")
        sel = df[df.artist.isin(top.index)]
        sel = (sel.groupby("artist", group_keys=False)
               .sample(n=cap, random_state=args.seed).reset_index(drop=True))
        sel["split"] = album_split(sel, args.test_folds, args.val_folds, args.seed)

        # Sanity checks: no album straddles splits, every artist is in every split.
        straddle = sel.groupby(["artist", "album"]).split.nunique().gt(1).sum()
        assert straddle == 0, f"{straddle} albums appear in more than one split"
        table = pd.crosstab(sel.artist, sel.split)
        if (table == 0).any().any():
            print("WARNING: some artists are missing from a split:\n", table[(table == 0).any(axis=1)])

        path = OUT / f"dataset_{tag}_cap{cap}.parquet"
        sel.to_parquet(path, index=False)
        report["dataset"] = {"path": str(path.relative_to(ROOT)), "artists": list(top.index),
                             "cap": cap, "splits": sel.split.value_counts().to_dict(),
                             "args": vars(args)}
        path.with_suffix(".json").write_text(json.dumps(report["dataset"], indent=2))
        share = table.div(table.sum(axis=1), axis=0)
        print(f"\nDataset: {len(top)} artists x {cap} songs -> {path.relative_to(ROOT)}")
        print(sel.split.value_counts().to_string())
        print("\nPer-artist split shares (min / median / max):")
        print(share.agg(["min", "median", "max"]).round(2).to_string())

    (OUT / "cleaning_report.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
