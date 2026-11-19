#!/usr/bin/env python3
"""Track city membership across k=3..10 in the existing GNN-all Ward tree.

This visualizes resolution changes, not resampling stability. Existing labels
are retained; each city's color stays fixed to its saved k=8 cluster.
"""

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, leaves_list, linkage
from scipy.spatial.distance import squareform

from .cluster_analysis import display_labels, transfer_similarity
from .plot_transfer_clusters import read_matrix

KS = tuple(range(3, 11))


def prepare(root):
    matrix = read_matrix(root, "gnn_all_pairs")
    ids = matrix.index.to_numpy()
    d = np.maximum(1 - transfer_similarity(matrix.loc[ids, ids].to_numpy()), 0)
    z = linkage(squareform(d), method="ward")
    order = leaves_list(z)
    rank = np.argsort(order)
    saved8 = (
        pd.read_csv(
            root / "results/gnn_all_pairs/k_8/cluster_membership.csv",
            dtype={"geoid": str},
        )
        .set_index("geoid")
        .loc[ids]
    )
    assert np.array_equal(rank, saved8.leaf_position)
    labels, positions, nodes, sizes = {}, {}, {}, {}
    records, transitions = [], []
    for k in KS:
        labels[k], _ = display_labels(fcluster(z, k, criterion="maxclust"), order, ids)
        assert len(np.unique(labels[k])) == k
        if k == 8:
            assert np.array_equal(labels[k], saved8.cluster)
        positions[k] = np.zeros(len(ids))
        sizes[k] = np.zeros(len(ids), dtype=int)
        nodes[k] = []
        group_order = list(dict.fromkeys(labels[k][order]))
        used = 0
        for group_index, cluster in enumerate(group_order):
            members = order[labels[k][order] == cluster]
            top = 0.8 * used / len(ids) + 0.2 * group_index / (k - 1)
            height = 0.8 * len(members) / len(ids)
            positions[k][members] = top + 0.8 * (np.arange(len(members)) + 0.5) / len(
                ids
            )
            sizes[k][members] = len(members)
            nodes[k].append(
                dict(cluster=int(cluster), top=top, height=height, n=len(members))
            )
            used += len(members)
            if k > KS[0]:
                parents = np.unique(labels[k - 1][members])
                assert len(parents) == 1, "Higher k must refine the existing hierarchy"
                transitions.append(
                    dict(
                        k_from=k - 1,
                        k_to=k,
                        cluster_from=int(parents[0]),
                        cluster_to=int(cluster),
                        n_msas=len(members),
                    )
                )
        assert used == len(ids) and np.all(np.diff(positions[k][order]) > 0)
        for index, gid in enumerate(ids):
            old_size = sizes[k - 1][index] if k > KS[0] else None
            new_size = int(sizes[k][index])
            records.append(
                dict(
                    geoid=gid,
                    name=saved8.loc[gid, "name"],
                    k=k,
                    cluster=int(labels[k][index]),
                    cluster_size=new_size,
                    cluster_k8=int(saved8.loc[gid, "cluster"]),
                    leaf_position=int(rank[index]),
                    y_position=float(positions[k][index]),
                    group_unchanged_from_previous_k=(
                        (new_size == old_size) if old_size is not None else None
                    ),
                    peer_retention_from_previous_k=(
                        ((new_size - 1) / (old_size - 1) if old_size > 1 else 1.0)
                        if old_size is not None
                        else np.nan
                    ),
                )
            )
    long = pd.DataFrame(records)
    output = root / ".work/gnn_all_pairs/cluster_tracking_k3_k10"
    output.mkdir(parents=True, exist_ok=True)
    long.to_csv(output / "city_membership.csv", index=False)
    wide = (
        saved8[["name", "cluster"]]
        .rename(columns={"cluster": "cluster_k8_reference"})
        .copy()
    )
    for k in KS:
        wide[f"cluster_k{k}"] = labels[k]
        wide[f"group_size_k{k}"] = sizes[k]
    wide["n_group_splits_k3_to_k10"] = sum(sizes[k] < sizes[k - 1] for k in KS[1:])
    wide.to_csv(output / "city_trajectories.csv", index_label="geoid")
    transitions = pd.DataFrame(transitions)
    assert transitions.groupby("k_from").n_msas.sum().eq(len(ids)).all()
    transitions.to_csv(output / "cluster_transitions.csv", index=False)
    return dict(
        ids=ids,
        order=order,
        saved8=saved8,
        labels=labels,
        positions=positions,
        nodes=nodes,
        sizes=sizes,
        output=output,
    )
