"""Build the modelling frame, the feature manifest and the split assignment.

Runs the as-of-application aggregation over all seven Home Credit tables in
DuckDB and writes four durable artifacts. Nothing here is computed in a notebook:
notebook 01 reads these files and narrates them.

Outputs, under ``<DATA_ROOT>/processed/adil/``:

``adil_frame.parquet``
    One row per ``SK_ID_CURR``, application columns plus every satellite
    aggregate.
``feature_manifest.parquet``
    One row per column, naming its source table, source column, aggregation and
    the as-of filter in force when it was computed.
``split_index.parquet``
    Train / calibration / test assignment.

and ``metrics/frame.json`` in the repository, which records the shape, the seed,
the per-split target rates and what every as-of filter removed.
"""

import json
import logging

import duckdb

from adil import features, paths, split

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger("build_features")


def main() -> None:
    processed = paths.processed_dir()
    con = duckdb.connect()
    con.execute("PRAGMA memory_limit='16GB'")
    # Single-threaded on purpose, and it costs about forty seconds.
    #
    # Floating-point addition is not associative, so a parallel SUM or MEAN
    # accumulates in whatever order the threads happen to finish and lands on a
    # slightly different value each run. Measured here: 48 of 559 numeric columns
    # differed bit-for-bit between two parallel builds, by around 1e-8. That is
    # far too small to see and quite large enough to move a LightGBM split
    # threshold, which moved test PR-AUC in the fifth decimal and changed every
    # report. A build that is reproducible is worth more than a build that is fast.
    con.execute("PRAGMA threads=1")

    log.info("counting what each as-of filter removes")
    filters = features.filter_report(con)
    for row in filters.itertuples():
        log.info(
            "  %-22s removed %10s of %12s rows (%.4f%%)",
            row.table,
            f"{row.rows_removed:,}",
            f"{row.rows_before:,}",
            100 * row.share_removed,
        )

    log.info("aggregating six satellite tables onto the applications")
    frame, manifest, skipped = features.build_frame(con)
    log.info("frame: %s rows x %s columns", f"{len(frame):,}", f"{frame.shape[1]:,}")
    for item in skipped:
        log.info("  categorical skipped for cardinality: %s", item)

    log.info("assigning the stratified split (seed %s)", split.SEED)
    assignment = split.stratified_split(frame)
    rates = assignment.groupby("split")["TARGET"].agg(["size", "mean"])
    for name, row in rates.iterrows():
        log.info("  %-12s n=%9s  target rate=%.5f", name, f"{int(row['size']):,}", row["mean"])

    frame.to_parquet(processed / "adil_frame.parquet", index=False)
    manifest.to_parquet(processed / "feature_manifest.parquet", index=False)
    assignment.to_parquet(processed / "split_index.parquet", index=False)
    filters.to_parquet(processed / "filter_report.parquet", index=False)

    unmanifested = sorted(set(frame.columns) - set(manifest["feature"]))
    phantom = sorted(set(manifest["feature"]) - set(frame.columns))
    summary = {
        "seed": split.SEED,
        "rows": int(len(frame)),
        "columns": int(frame.shape[1]),
        "manifest_rows": int(len(manifest)),
        "columns_without_a_manifest_row": unmanifested,
        "manifest_rows_without_a_column": phantom,
        "target_rate_overall": float(frame["TARGET"].mean()),
        "target_rate_by_split": {name: float(value) for name, value in rates["mean"].items()},
        "target_rate_spread": float(rates["mean"].max() - rates["mean"].min()),
        "rows_by_split": {name: int(value) for name, value in rates["size"].items()},
        "categoricals_skipped_for_cardinality": skipped,
        "as_of_filters": filters.to_dict("records"),
    }
    (paths.metrics_dir() / "frame.json").write_text(json.dumps(summary, indent=2) + "\n")

    # Checked in both directions. A column with no manifest row cannot prove its
    # provenance; a manifest row with no column means the manifest describes a frame
    # that does not exist, which is the more dangerous of the two because it reads as
    # a complete audit trail.
    if unmanifested:
        raise SystemExit(
            f"{len(unmanifested)} columns have no manifest row and cannot prove their "
            f"provenance: {unmanifested[:10]}"
        )
    if phantom:
        raise SystemExit(
            f"{len(phantom)} manifest rows describe columns that are not in the frame: "
            f"{phantom[:10]}"
        )
    log.info("done — every one of %s columns has a manifest row", f"{frame.shape[1]:,}")


if __name__ == "__main__":
    main()
