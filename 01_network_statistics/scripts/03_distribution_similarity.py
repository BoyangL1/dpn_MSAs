#!/usr/bin/env python3
"""Compare metropolitan strength and dependency-weight distributions."""
from pathlib import Path
import argparse
import json

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon

ROOT = Path(__file__).resolve().parents[1]


def run(root):
    work_dir = root / ".work"
    nodes = pd.read_parquet(
        work_dir / "network_nodes.parquet", columns=["GEOID", "in_str", "out_str"]
    )
    edges = pd.read_parquet(
        work_dir / "network_edges.parquet",
        columns=["WIJ", "source_msa", "is_intra_msa"],
    )
    msa = pd.read_csv(work_dir / "msa_summary.csv", dtype={"GEOID": str})
    ids = sorted(msa.GEOID)
    nodes["GEOID"] = nodes.GEOID.astype(str)
    groups = nodes.groupby("GEOID", sort=True)
    samples = {
        name: [groups.get_group(g)[column].to_numpy() for g in ids]
        for name, column in [("in_strength", "in_str"), ("out_strength", "out_str")]
    }
    weights = edges.loc[edges.is_intra_msa].groupby("source_msa", sort=True).WIJ
    samples["edge_weight"] = [weights.get_group(g).to_numpy() for g in ids]
    pair_i, pair_j = np.triu_indices(len(ids), 1)
    rows = []
    for metric, values in samples.items():
        positive = [v[v > 0] for v in values]
        if not all(len(v) for v in positive):
            raise ValueError(f"Empty positive support: {metric}")
        bounds = (
            (-4.0, np.log10(600.0))
            if metric != "edge_weight"
            else (
                float(np.floor(np.log10(min(v.min() for v in positive)))),
                float(np.ceil(np.log10(max(v.max() for v in positive)))),
            )
        )
        for n_bins in (25, 49, 98):
            bins = np.logspace(*bounds, n_bins + 1)
            counts = np.array([np.histogram(v, bins=bins)[0] for v in positive])
            if not np.array_equal(counts.sum(axis=1), list(map(len, positive))):
                raise ValueError(f"Histogram support excludes observations: {metric}")
            probability = counts / counts.sum(axis=1, keepdims=True)
            distances = jensenshannon(
                probability[pair_i], probability[pair_j], base=2, axis=1
            )
            q25, median, q75 = np.quantile(distances, [0.25, 0.5, 0.75])
            rows.append(
                dict(
                    metric=metric,
                    n_bins=n_bins,
                    n_msa=len(ids),
                    n_pairs=len(distances),
                    median_js_distance=median,
                    q25=q25,
                    q75=q75,
                    mean_js_distance=distances.mean(),
                )
            )
    summary = pd.DataFrame(rows)
    summary.to_csv(work_dir / "distribution_js_summary.csv", index=False)
    notes = {
        "definition": "Base-2 Jensen-Shannon distance; square root of divergence.",
        "primary_bins": 49,
        "sensitivity_bins": [25, 98],
        "construction": "Shared logarithmic bins; probability mass; no smoothing; positive values only.",
        "aggregation": "Equal weight for every unordered pair of distinct metropolitan areas.",
        "scope": "National in/out strengths include cross-metropolitan edges; edge weights use internal edges.",
    }
    (work_dir / "distribution_js_notes.json").write_text(
        json.dumps(notes, indent=2) + "\n"
    )
    print(summary.to_string(index=False))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    run(parser.parse_args().root.resolve())
