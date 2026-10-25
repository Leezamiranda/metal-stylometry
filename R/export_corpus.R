# Export text2map.corpora's corpus_metal_lyrics to Parquet for the Python pipeline.
#
# Usage (from the project root):
#   Rscript R/export_corpus.R            # writes data/raw/metal_lyrics.parquet
#   Rscript R/export_corpus.R --csv      # also writes a CSV copy (large, slow)
#
# This script only mirrors the corpus to disk. All cleaning happens in Python
# (src/prepare_data.py), so the raw export stays a faithful copy of the source.

suppressPackageStartupMessages({
  library(text2map.corpora)
  library(nanoparquet)
})

args <- commandArgs(trailingOnly = TRUE)
out_dir <- file.path("data", "raw")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

# download_corpus() caches the file inside the package's data directory;
# skip the download if it's already there.
cached <- system.file("data", "corpus_metal_lyrics.qs2", package = "text2map.corpora")
if (!nzchar(cached)) {
  download_corpus("corpus_metal_lyrics")
}
df <- load_corpus("corpus_metal_lyrics")

# Sanity checks: fail loudly if the upstream corpus changes shape.
expected_cols <- c(
  "song_number", "song_title", "song_lyrics_url", "artist", "album", "lyrics",
  "year", "score", "subgenre", "score_number", "genre", "language",
  "lyrics_prepped", "title_prepped", "word_position"
)
missing <- setdiff(expected_cols, names(df))
if (length(missing)) stop("Missing columns: ", paste(missing, collapse = ", "))

# The corpus has no unique ID: song_number is the track number within an album,
# and albums tagged with several genres appear once per genre (same URL, same
# lyrics, different genre/score). Add a stable row ID so Python-side cleaning
# decisions can be traced back to the source rows.
df <- cbind(row_id = seq_len(nrow(df)), df)
message(sprintf("%s rows, %s unique lyrics URLs",
                format(nrow(df), big.mark = ","),
                format(length(unique(df$song_lyrics_url)), big.mark = ",")))

parquet_path <- file.path(out_dir, "metal_lyrics.parquet")
write_parquet(df, parquet_path, compression = "zstd")
message(sprintf("Wrote %s rows x %s cols to %s (%.1f MB)",
                format(nrow(df), big.mark = ","), ncol(df), parquet_path,
                file.size(parquet_path) / 1e6))

if ("--csv" %in% args) {
  csv_path <- file.path(out_dir, "metal_lyrics.csv")
  # write.csv quotes every string field and doubles embedded quotes (RFC 4180),
  # which pandas.read_csv parses correctly. UTF-8 is preserved.
  write.csv(df, csv_path, row.names = FALSE, fileEncoding = "UTF-8", na = "")
  message(sprintf("Wrote %s (%.1f MB)", csv_path, file.size(csv_path) / 1e6))
}
