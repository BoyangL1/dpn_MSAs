"""Ward clustering of direct mean-bidirectional transfer dissimilarity."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, leaves_list, linkage
from scipy.spatial.distance import squareform

RESOLUTIONS = (8,)
REPRESENTATION = (
    "historical_direct_transfer_dissimilarity_one_minus_mean_bidirectional_r"
)
RESOLUTION_RULE = "fixed_descriptive_resolution_8"
LABEL_RULE = "new_york_C2_state_college_C6_if_distinct_and_available_other_labels_by_leaf_order_swaps"
LABEL_ANCHORS = (("35620", 2), ("44300", 6))


def transfer_similarity(values):
    """Return historical S = (T + T.T) / 2 with its diagonal set to one.

    Observed self scores do not affect clustering. No row-feature distances,
    centering, scaling, or residualization are used. The input T is retained.
    """
    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or values.shape[0] != values.shape[1] or len(values) < 3:
        raise ValueError("Expected a square transfer matrix for at least three MSAs")
    if not np.isfinite(values).all() or np.max(np.abs(values)) > 1 + 1e-6:
        raise ValueError("Transfer correlations must be finite and within [-1, 1]")
    similarity = (values + values.T) / 2.0
    np.fill_diagonal(similarity, 1.0)
    return similarity


def leaf_labels(raw, order):
    """Assign contiguous display labels independently within one hierarchy."""
    first_seen = list(dict.fromkeys(map(int, np.asarray(raw)[order])))
    mapping = {value: i + 1 for i, value in enumerate(first_seen)}
    return np.array([mapping[int(value)] for value in raw], dtype=int), mapping


def display_labels(raw, order, ids):
    """Pin existing groups to display labels without changing membership/order.

    Start with leaf-order labels, then swap New York's group into C2 and State
    College's group into C6 (when k >= 6). Missing anchors are skipped, allowing
    synthetic or restricted cohorts. If the anchors share a cluster, New York
    takes precedence and the State College pin is skipped; membership is never split.
    """
    raw = np.asarray(raw)
    ids = np.asarray(ids).astype(str)
    if ids.shape != raw.shape or len(np.unique(ids)) != len(ids):
        raise ValueError("Unique city IDs must align with raw cluster labels")
    _, mapping = leaf_labels(raw, order)
    pins = {}
    for geoid, target in LABEL_ANCHORS:
        matches = np.flatnonzero(ids == geoid)
        if target > len(mapping) or not len(matches):
            continue
        anchor = int(raw[matches[0]])
        if anchor in pins and pins[anchor] != target:
            continue
        pins[anchor] = target
    for anchor, target in pins.items():
        displaced = next(key for key, value in mapping.items() if value == target)
        mapping[displaced], mapping[anchor] = mapping[anchor], target
    return np.array([mapping[int(value)] for value in raw], dtype=int), mapping


def case_at_k(case, k):
    """Expose one fixed descriptive cut without changing the base hierarchy."""
    if int(k) != k or k not in case["resolutions"]:
        raise ValueError("The primary descriptive resolution is k=8")
    k = int(k)
    raw = fcluster(case["z"], k, criterion="maxclust")
    if len(np.unique(raw)) != k:
        raise ValueError(f"The hierarchy cannot produce exactly {k} distinct clusters")
    labels, mapping = display_labels(raw, case["order"], case["ids"])
    return dict(case, raw=raw, labels=labels, mapping=mapping, k=k)


def analyze_matrix(frame, ids=None):
    """Construct the fixed Ward hierarchy without resolution or stability sweeps."""
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("Expected a DataFrame with matching row and column MSA IDs")
    if not frame.index.is_unique or not frame.columns.is_unique:
        raise ValueError("Both matrix axes must contain unique MSA IDs")
    if set(frame.index) != set(frame.columns):
        raise ValueError("Matrix row and column MSA IDs must match")
    ids = frame.index.to_numpy() if ids is None else np.asarray(ids)
    if not pd.Index(ids).is_unique:
        raise ValueError("MSA IDs must be unique")
    values = frame.loc[ids, ids].to_numpy(dtype=float)
    similarity = transfer_similarity(values)
    distance = np.maximum(1.0 - similarity, 0.0)
    np.fill_diagonal(distance, 0.0)
    z = linkage(squareform(distance, checks=True), method="ward")
    return dict(
        ids=ids,
        values=values,
        similarity=similarity,
        distance=distance,
        z=z,
        order=leaves_list(z),
        resolutions=RESOLUTIONS,
        representation=REPRESENTATION,
        label_rule=LABEL_RULE,
    )
