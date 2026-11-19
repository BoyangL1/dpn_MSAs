#!/usr/bin/env python3
"""Select the GNN analysis cohort using validation-selected best_epoch > 1."""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
MIN_DELTA = 1e-5
COHORT_RULE = "best_epoch_gt_1"
COHORT_DESCRIPTION = "Exclude validation-selected best_epoch == 1"


def retained_ids(frame):
    """Check the recorded cohort against the validation-epoch rule."""
    expected = frame.best_epoch.gt(1)
    actual = frame.retained_for_clustering.astype(str).str.lower().eq("true")
    if not actual.equals(expected):
        raise ValueError("Run 03_select_cohort.py to refresh the analysis cohort")
    return np.asarray(sorted(frame.index[expected]))


def select_cohort(root):
    root = Path(root).resolve()
    work = root / ".work"
    path = work / "within_msa_performance.csv"
    frame = pd.read_csv(path, dtype={"geoid": str})
    assert frame.geoid.is_unique and len(frame) > 0
    epochs = frame[["best_epoch", "epochs_run"]].to_numpy()
    assert np.isfinite(epochs).all() and (epochs == np.floor(epochs)).all()
    assert frame.best_epoch.ge(1).all() and frame.best_epoch.le(frame.epochs_run).all()
    frame["improved_after_first_epoch"] = frame.best_epoch.gt(1)
    frame["retained_for_clustering"] = frame.best_epoch.gt(1)
    frame["validation_min_delta"] = MIN_DELTA
    frame["retention_reason"] = np.where(
        frame.best_epoch.gt(1), "retained_best_epoch_gt_1", "excluded_best_epoch_eq_1"
    )
    frame.to_csv(path, index=False)
    ids = retained_ids(frame.set_index("geoid"))
    matrix = pd.read_csv(work / "gnn_all_pairs/transfer_pearson.csv", index_col=0)
    matrix.index, matrix.columns = matrix.index.astype(str), matrix.columns.astype(str)
    if not set(ids).issubset(matrix.index) or not set(ids).issubset(matrix.columns):
        raise ValueError("The transfer matrix is missing analysis-cohort cities")
    output = root / "results/gnn_all_pairs"
    output.mkdir(parents=True, exist_ok=True)
    matrix.loc[ids, ids].to_csv(output / "transfer_pearson_analysis.csv")
    settings_path = work / "training_settings.csv"
    if settings_path.exists():
        settings = pd.read_csv(settings_path, dtype=str).set_index("setting")["value"]
        updates = {
            "clustering_retention_rule": COHORT_DESCRIPTION,
            "n_retained_clustering_msas": str(len(ids)),
            "n_excluded_clustering_msas": str(len(frame) - len(ids)),
            "cohort_rule": COHORT_RULE,
            "validation_min_delta": str(MIN_DELTA),
        }
        for key, value in updates.items():
            settings.loc[key] = value
        settings.rename("value").rename_axis("setting").reset_index().to_csv(
            settings_path, index=False
        )
    print(f"Primary cohort: {len(ids)}/{len(frame)} cities retained", flush=True)
    return frame


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    select_cohort(parser.parse_args().root.resolve())


if __name__ == "__main__":
    main()
