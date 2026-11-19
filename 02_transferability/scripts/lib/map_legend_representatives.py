#!/usr/bin/env python3
"""Configured map examples with recorded selection rules and centrality.

Familiarity and contextual relevance are editorial judgments, not estimated
scores. Per-map node-count rules and explicit per-cluster user choices are supported.
Distance ranks are diagnostics and do not affect largest-node-count selection.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

CONFIG = "config/map_legend_representatives.json"
METHOD = "recognizable_examples_ranked_by_transfer_and_feature_centrality"
NODE_COUNT_METHOD = "largest_node_count_within_cluster"
USER_METHOD = "user_selected_illustrative_examples"


def membership_fingerprint(ids, labels):
    pairs = sorted(zip(map(str, ids), map(int, labels)))
    return hashlib.sha256(
        "".join(f"{gid}:{label}\n" for gid, label in pairs).encode()
    ).hexdigest()


def curated_map_examples(root, view, case, meta):
    path = Path(root) / CONFIG
    if not path.exists():
        return None
    config = json.loads(path.read_text())
    choice = config["selections"].get(f'{view}/k_{case["k"]}')
    if choice is None:
        return None
    method = choice.get("method", METHOD)
    if method not in (METHOD, NODE_COUNT_METHOD):
        raise ValueError(f"Unknown map example selection method: {method}")
    ids = np.asarray(case["ids"]).astype(str)
    labels = np.asarray(case["labels"])
    if choice["membership_sha256"] != membership_fingerprint(ids, labels):
        raise ValueError(
            "Curated map examples need review: the cluster memberships changed"
        )
    if set(choice["clusters"]) != {str(c) for c in range(1, case["k"] + 1)}:
        raise ValueError("Curated examples must cover every cluster")
    rows = []
    for c in range(1, case["k"] + 1):
        cluster_method = choice.get("cluster_methods", {}).get(str(c), method)
        if cluster_method not in (METHOD, NODE_COUNT_METHOD, USER_METHOD):
            raise ValueError(f"C{c}: unknown selection method {cluster_method}")
        mask = labels == c
        group_ids = ids[mask]
        choices = choice["clusters"][str(c)]
        selected = [item["geoid"] for item in choices]
        if len(selected) != min(3, len(group_ids)) or len(set(selected)) != len(
            selected
        ):
            raise ValueError(f"C{c}: expected up to three distinct examples")
        if not set(selected) <= set(group_ids):
            raise ValueError(f"C{c}: examples must belong to this cluster")
        if cluster_method == NODE_COUNT_METHOD:
            expected = sorted(
                group_ids, key=lambda gid: (-int(meta.loc[gid, "n_nodes"]), gid)
            )[:3]
            if selected != expected:
                raise ValueError(
                    f"C{c}: configured examples must be the three largest MSAs by node count"
                )
        block = case["distance"][np.ix_(mask, mask)]
        means = (
            block.sum(axis=1) / (len(group_ids) - 1)
            if len(group_ids) > 1
            else np.full(1, np.nan)
        )
        scores = pd.Series(means, index=group_ids)
        ranks = scores.rank(method="min")
        for rank, item in enumerate(choices, 1):
            gid = item["geoid"]
            if not item["reason"].strip():
                raise ValueError("Each curated city needs a selection reason")
            rows.append(
                dict(
                    cluster=c,
                    geoid=gid,
                    name=meta.loc[gid, "name"],
                    n_nodes=int(meta.loc[gid, "n_nodes"]),
                    rank=rank,
                    selection_method=cluster_method,
                    selection_reason=item["reason"],
                    mean_transfer_dissimilarity=float(scores.loc[gid]),
                    within_cluster_distance_rank=float(ranks.loc[gid]),
                    n_cluster_msas=int(mask.sum()),
                )
            )
    return pd.DataFrame(rows)
