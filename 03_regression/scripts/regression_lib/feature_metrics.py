"""Network and urban-form descriptors."""

from collections import defaultdict

import numpy as np

import pandas as pd

import scipy.sparse as sp

from scipy import ndimage

import networkx as nx

import igraph as ig

CONFIGS = [("main", 0.25, 1)]

ALPHA = 0.85

TOL = 1e-12

MIN_CENTER_POI = 50

MIN_CENTER_CELLS = 3

QUEEN = np.ones((3, 3), dtype=int)


def describe_graph(s, t, w, n_total, configs=CONFIGS):
    weights = defaultdict(float)
    for a, b, weight in zip(s, t, w):
        if a != b:
            weights[(a, b) if a < b else (b, a)] += weight
    graph = nx.Graph()
    graph.add_weighted_edges_from((a, b, wt) for (a, b), wt in weights.items())
    if graph.number_of_nodes() < 3 or n_total < graph.number_of_nodes():
        raise ValueError("Invalid node universe or empty network")
    components = list(nx.connected_components(graph))
    giant = graph.subgraph(max(components, key=len))
    degree = np.zeros(n_total, dtype=float)
    degree[: graph.number_of_nodes()] = [d for _, d in graph.degree()]
    degree.sort()
    degree_gini = (
        2 * np.dot(np.arange(1, n_total + 1), degree) / (n_total * degree.sum())
        - (n_total + 1) / n_total
    )
    idx = {node: i for i, node in enumerate(graph)}
    simple = ig.Graph(
        n=n_total, edges=[(idx[a], idx[b]) for a, b in graph.edges()], directed=False
    )
    transitivity = simple.transitivity_undirected(mode="zero")
    base = dict(
        n_nodes=n_total,
        active_nodes=graph.number_of_nodes(),
        n_undirected_edges=graph.number_of_edges(),
        giant_nodes=len(giant),
        net_degree_gini=degree_gini,
        net_transitivity=transitivity,
        net_giant_component_share=len(giant) / n_total,
    )
    rows = []
    for label, resolution, seed in configs:
        comms = sorted(
            nx.community.louvain_communities(
                giant, weight="weight", resolution=resolution, seed=seed
            ),
            key=len,
            reverse=True,
        )
        shares = np.array([len(c) / len(giant) for c in comms])
        rows.append(
            dict(
                base,
                configuration=label,
                resolution=resolution,
                seed=seed,
                n_communities=len(comms),
                net_largest_community_share=shares.max(),
                community_entropy=float(-(shares * np.log(shares)).sum()),
                net_modularity=nx.community.modularity(
                    giant, comms, weight="weight", resolution=resolution
                ),
            )
        )
    return rows


def pagerank(w, alpha=ALPHA):
    """Global L1 tolerance (not N-scaled); uniform teleport and dangling redistribution."""
    n = w.shape[0]
    rows = np.asarray(w.sum(axis=1)).ravel()
    inv = np.divide(1.0, rows, out=np.zeros(n), where=rows > 0)
    pt = (sp.diags(inv) @ w).T.tocsr()
    dangling = rows == 0
    p = np.full(n, 1.0 / n)
    for iteration in range(1, 2001):
        nxt = alpha * (pt @ p + p[dangling].sum() / n) + (1 - alpha) / n
        error = float(np.abs(nxt - p).sum())
        p = nxt
        if error < TOL:
            residual = float(
                np.abs(
                    alpha * (pt @ p + p[dangling].sum() / n) + (1 - alpha) / n - p
                ).sum()
            )
            assert np.isfinite(p).all() and (p >= 0).all() and abs(p.sum() - 1) < 1e-10
            return p, iteration, residual
    raise RuntimeError("PageRank did not converge")


def gini(x):
    x = np.sort(np.asarray(x, dtype=float))
    return float(
        2 * np.dot(np.arange(1, len(x) + 1), x) / (len(x) * x.sum())
        - (len(x) + 1) / len(x)
    )


def metrics_for_msa(sub_cells: pd.DataFrame, total_poi: int, top_frac: float) -> dict:
    """sub_cells: one row per occupied grid cell with columns [ix, iy, n]."""
    out = dict(
        total_poi=int(total_poi),
        n_centers=0,
        largest_center_share=np.nan,
        effective_centers=np.nan,
        largest_center_poi=0,
        center_poi_total=0,
        share_in_centers=0.0,
    )

    n_all = sub_cells["n"].to_numpy()
    if n_all.size == 0:
        return out

    thr = np.quantile(n_all, 1.0 - top_frac)
    cand = sub_cells[sub_cells["n"] >= thr]
    if len(cand) == 0:
        return out

    ix = cand["ix"].to_numpy()
    iy = cand["iy"].to_numpy()
    cnt = cand["n"].to_numpy().astype(np.int64)
    ix0, iy0 = ix.min(), iy.min()
    nx, ny = ix.max() - ix0 + 1, iy.max() - iy0 + 1

    mask = np.zeros((nx, ny), dtype=bool)
    cgrid = np.zeros((nx, ny), dtype=np.int64)
    mask[ix - ix0, iy - iy0] = True
    cgrid[ix - ix0, iy - iy0] = cnt

    labels, k = ndimage.label(mask, structure=QUEEN)
    if k == 0:
        return out

    idx = np.arange(1, k + 1)
    center_poi = ndimage.sum_labels(cgrid, labels, index=idx).astype(np.int64)
    center_cells = np.bincount(labels.ravel())[1 : k + 1]

    keep = (center_poi >= MIN_CENTER_POI) & (center_cells >= MIN_CENTER_CELLS)
    S = center_poi[keep].astype(np.float64)
    if S.size == 0:
        return out

    p = S / S.sum()
    out["n_centers"] = int(S.size)
    out["largest_center_share"] = float(p.max())
    out["effective_centers"] = float(1.0 / np.sum(p**2))
    out["largest_center_poi"] = int(S.max())
    out["center_poi_total"] = int(S.sum())
    out["share_in_centers"] = float(S.sum() / total_poi) if total_poi else 0.0
    return out
