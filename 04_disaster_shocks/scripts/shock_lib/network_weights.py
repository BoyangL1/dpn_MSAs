"""Attach original directional dependency weights and visitation counts."""

import duckdb
import numpy as np
import pandas as pd

SOURCE_NAME = "us_dependency_network_20210101_20211231_new_nij_gt50_cleaned.csv"
WEIGHT_DEFINITION = "W[source,target] = original CSV WIJ = NIJ / NI_SOURCE; NI_SOURCE is source-node visitation. Within-MSA positive edges are retained without row normalization or spectral rescaling."


def attach_original_weights(nodes, edges, source):
    """Join source data by PLACEKEY pairs while preserving every frozen edge row."""
    assert nodes.idx.is_unique and nodes.PLACEKEY.is_unique
    assert not edges.duplicated(["src_idx", "tgt_idx"]).any()
    frozen = edges[["src_idx", "tgt_idx", "nij"]].copy()
    frozen["edge_row"] = np.arange(len(frozen), dtype=np.int64)
    con = duckdb.connect()
    con.execute("SET threads=4")
    con.execute("SET memory_limit='8GB'")
    con.register("node_keys", nodes[["idx", "PLACEKEY"]])
    con.register("frozen_edges", frozen)
    matched = con.execute(
        "\n        WITH source_edges AS (\n            SELECT s.idx AS src_idx, t.idx AS tgt_idx,\n                   raw.NIJ AS original_nij, raw.NI_SOURCE AS ni_source,\n                   raw.WIJ AS wij\n            FROM read_csv(?, header=true, columns={\n                'SOURCE':'VARCHAR', 'TARGET':'VARCHAR', 'NIJ':'DOUBLE',\n                'NI_SOURCE':'DOUBLE', 'NJ_TARGET':'DOUBLE', 'WIJ':'DOUBLE'}) raw\n            JOIN node_keys s ON raw.SOURCE=s.PLACEKEY\n            JOIN node_keys t ON raw.TARGET=t.PLACEKEY\n        )\n        SELECT f.edge_row, r.original_nij, r.ni_source, r.wij\n        FROM frozen_edges f LEFT JOIN source_edges r\n        ON f.src_idx=r.src_idx AND f.tgt_idx=r.tgt_idx\n        ORDER BY f.edge_row\n    ",
        [str(source)],
    ).fetchdf()
    con.close()
    assert len(matched) == len(
        edges
    ), "Source pairs do not match frozen edges one-to-one"
    assert np.array_equal(matched.edge_row.to_numpy(), frozen.edge_row.to_numpy())
    assert np.isfinite(matched[["original_nij", "ni_source", "wij"]]).all().all()
    assert np.array_equal(matched.original_nij.to_numpy(), edges.nij.to_numpy())
    assert matched.ni_source.gt(0).all()
    assert matched.wij.between(0, 1).all(), "Original WIJ outside [0, 1]; do not clip"
    expected = matched.original_nij.to_numpy() / matched.ni_source.to_numpy()
    error = np.abs(matched.wij.to_numpy() - expected)
    assert error.max(initial=0.0) < 1e-12, "Original WIJ differs from NIJ / NI_SOURCE"
    by_source = (
        pd.DataFrame(
            {
                "src_idx": edges.src_idx.to_numpy(),
                "ni_source": matched.ni_source.to_numpy(),
            }
        )
        .groupby("src_idx")
        .ni_source
    )
    assert by_source.nunique().eq(1).all(), "NI_SOURCE varies for a source node"
    restored = edges.copy()
    restored["wij"] = matched.wij.to_numpy()
    restored["ni_source"] = matched.ni_source.to_numpy()
    for field in ["src_idx", "tgt_idx", "nij", "msa_geoid"]:
        assert restored[field].equals(edges[field]), field
    return restored


def attach_node_visits(nodes, source):
    """Verify the same visitation total on all source and target occurrences."""
    con = duckdb.connect()
    con.execute("SET threads=4")
    con.execute("SET memory_limit='8GB'")
    con.register("node_keys", nodes[["idx", "PLACEKEY"]])
    visits = con.execute(
        "\n        WITH raw AS MATERIALIZED (\n            SELECT SOURCE, TARGET, NI_SOURCE, NJ_TARGET\n            FROM read_csv(?, header=true, columns={\n                'SOURCE':'VARCHAR', 'TARGET':'VARCHAR', 'NIJ':'DOUBLE',\n                'NI_SOURCE':'DOUBLE', 'NJ_TARGET':'DOUBLE', 'WIJ':'DOUBLE'})\n        ), node_values AS (\n            SELECT n.idx, r.NI_SOURCE AS visits\n            FROM raw r JOIN node_keys n ON r.SOURCE=n.PLACEKEY\n            UNION ALL\n            SELECT n.idx, r.NJ_TARGET AS visits\n            FROM raw r JOIN node_keys n ON r.TARGET=n.PLACEKEY\n        )\n        SELECT idx, min(visits) AS min_visits, max(visits) AS max_visits,\n               count(*) AS n_occurrences, count(visits) AS n_nonnull\n        FROM node_values GROUP BY idx ORDER BY idx\n    ",
        [str(source)],
    ).fetchdf()
    con.close()
    assert len(visits) == len(
        nodes
    ), "Some frozen nodes have no original visitation count"
    assert visits.idx.is_unique and set(visits.idx) == set(nodes.idx)
    assert visits.n_occurrences.eq(visits.n_nonnull).all()
    assert np.isfinite(visits[["min_visits", "max_visits"]]).all().all()
    assert visits.min_visits.gt(0).all()
    conflicts = visits.min_visits.ne(visits.max_visits)
    assert (
        not conflicts.any()
    ), f"Inconsistent source/target visitation: {visits.loc[conflicts].head().to_dict('records')}"
    restored = nodes.copy()
    restored["ni_source"] = restored.idx.map(visits.set_index("idx").min_visits)
    for field in nodes.columns:
        if field != "ni_source":
            assert restored[field].equals(nodes[field]), field
    return restored
