"""Split unordered POI pairs within each MSA."""

import numpy as np

SPLIT = (0.70, 0.10, 0.20)


def split_pairs_per_msa(pairs, n_nodes, rng):
    split = np.full(len(pairs), "test", dtype=object)
    counts = {name: 0 for name in ("train", "val", "test")}
    if not len(pairs):
        return split, counts
    msa = pairs.msa_geoid.to_numpy()
    src = pairs.src_idx.to_numpy(np.int64)
    tgt = pairs.tgt_idx.to_numpy(np.int64)
    order = np.argsort(msa, kind="stable")
    ordered = msa[order]
    bounds = np.flatnonzero(np.r_[True, ordered[1:] != ordered[:-1], True])
    for left, right in zip(bounds[:-1], bounds[1:]):
        rows = order[left:right]
        keys = np.minimum(src[rows], tgt[rows]) * np.int64(n_nodes) + np.maximum(
            src[rows], tgt[rows]
        )
        unique, inverse = np.unique(keys, return_inverse=True)
        shuffled = rng.permutation(len(unique))
        nt, nv = int(SPLIT[0] * len(unique)), int(SPLIT[1] * len(unique))
        pair_labels = np.full(len(unique), "test", dtype=object)
        pair_labels[shuffled[:nt]] = "train"
        pair_labels[shuffled[nt : nt + nv]] = "val"
        assigned = pair_labels[inverse]
        split[rows] = assigned
        for label in counts:
            counts[label] += int(np.count_nonzero(assigned == label))
    return split, counts
