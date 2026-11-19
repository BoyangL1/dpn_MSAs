#!/usr/bin/env python3
"""Single-layer BiSAGE transfer with an internal gravity baseline."""
from __future__ import annotations
import argparse
import ast
import hashlib
import json
import random
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import LinearRegression

ROOT = Path(__file__).resolve().parents[2]
BUILD = ROOT / ".work/build"
MSA_SUMMARY = BUILD / "msa_summary.csv"
TRAINING_RECORD_KEYS = (
    "base_coefs",
    "best_val",
    "best_epoch",
    "epochs_run",
    "train_seconds",
    "max_epochs",
    "stop_reason",
    "epochs_without_improvement",
    "history",
)


def configure_paths(root):
    global ROOT, BUILD, MSA_SUMMARY
    ROOT = Path(root).resolve()
    BUILD = ROOT / ".work/build"
    MSA_SUMMARY = BUILD / "msa_summary.csv"


def short_msa_name(name):
    if not isinstance(name, str):
        return str(name)
    if "," in name:
        city, state = name.split(",", 1)
        return f"{city.split('-')[0].strip()} {state.strip().split('-')[0].strip()}"
    return name[:18]


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def training_protocol_fingerprint(source=None):
    """Hash numerical code while excluding provenance and final CSV reporting.

    Imports, globals, model functions, and the training/evaluation loop are
    included; final CSV reporting is excluded.
    """
    tree = ast.parse(Path(__file__).read_text() if source is None else source)
    excluded = {"training_protocol_fingerprint", "resolve_run_signature"}
    keep = []
    for node in tree.body:
        if (
            isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
        ):
            continue
        if isinstance(node, ast.Import) and all(
            item.name == "ast" for item in node.names
        ):
            continue
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in excluded:
            continue
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id.startswith("LEGACY_") for t in node.targets
        ):
            continue
        if isinstance(node, ast.FunctionDef) and node.name == "main":

            def at(name):
                return next(
                    i
                    for i, statement in enumerate(node.body)
                    if isinstance(statement, ast.Assign)
                    and any(
                        isinstance(t, ast.Name) and t.id == name
                        for t in statement.targets
                    )
                )

            node.body = (
                node.body[: at("manifest_args")] + node.body[at("rows") : at("metrics")]
            )
        keep.append(node)
    canonical = ast.dump(
        ast.Module(body=keep, type_ignores=[]), include_attributes=False
    )
    return hashlib.sha256(canonical.encode()).hexdigest()


def resolve_run_signature(manifest, model_dir):
    """Resume only the exact saved data, settings and model-source version."""
    manifest["executed_script_sha256"] = manifest["script_sha256"]
    manifest["training_protocol_sha256"] = training_protocol_fingerprint()
    keys = (
        "args",
        "top_msas",
        "build_table_sha256",
        "torch_version",
        "cuda_version",
        "script_sha256",
        "architecture",
        "model_code_sha256",
    )
    payload = {key: manifest[key] for key in keys}
    signature = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    previous = Path(model_dir) / "preprocessing.pt"
    if previous.exists():
        saved = torch.load(previous, map_location="cpu", weights_only=True)
        if saved["signature"] != signature:
            raise RuntimeError(
                "Existing models do not match the current numerical protocol, data or settings; "
                "use a fresh --root for a new experiment"
            )
    return signature


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--top-msas", type=int, default=376)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch", type=int, default=16384)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--warmup-epochs", type=int, default=2)
    parser.add_argument("--weight-pow", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)
    if args.epochs < 1:
        parser.error("--epochs must be positive")
    args.architecture, args.layers, args.dropout = "legacy", 1, 0.0
    try:
        architecture_config(args)
    except ValueError as exc:
        parser.error(str(exc))

    args.no_name_emb = args.no_cat_emb = args.no_cat_id = True
    args.bge_proj_dim, args.cat_emb_dim = 64, 16
    args.zero_shot_no_graph = False
    return args


def architecture_config(args):
    """Explicit checkpoint schema; absent architecture fields mean legacy."""
    name = getattr(args, "architecture", "legacy")
    config = dict(
        name=name,
        layers=getattr(args, "layers", 1),
        hidden=args.hidden,
        dropout=getattr(args, "dropout", 0.0),
        edge_dim=4,
    )
    if name != "legacy":
        raise ValueError(f"Unknown encoder architecture: {name}")
    if config["layers"] != 1 or config["hidden"] < 1:
        raise ValueError("BiSAGE requires one layer and a positive hidden size")
    if not np.isfinite(config["dropout"]) or not 0 <= config["dropout"] < 1:
        raise ValueError("Dropout must be finite and in [0, 1)")
    if name == "legacy" and (config["layers"] != 1 or config["dropout"] != 0):
        raise ValueError("BiSAGE requires one layer and zero dropout")
    return config


def checkpoint_architecture(state):
    """Read saved model settings without depending on today's CLI defaults."""
    config = state.get("architecture")
    if config is None:
        config = dict(
            name="legacy", layers=1, hidden=state["hidden"], dropout=0.0, edge_dim=4
        )
    args = argparse.Namespace(
        architecture=config["name"],
        layers=config["layers"],
        hidden=config["hidden"],
        dropout=config["dropout"],
    )
    checked = architecture_config(args)
    if config != checked or state.get("hidden", checked["hidden"]) != checked["hidden"]:
        raise ValueError("Inconsistent saved encoder architecture")
    if state.get("edge_dim", 4) != 4:
        raise ValueError("Checkpoint must use the existing four pair features")
    return checked


def make_encoder(config, in_dim):
    """Construct the single-layer BiSAGE encoder."""
    config = checkpoint_architecture({"architecture": config})
    return BiSAGE(in_dim, config["hidden"], layers=1)


def check_existing_architecture(root, args):
    """Fail before building data or writing outputs when a run would mix models."""
    model_dir = Path(root) / "models"
    path = model_dir / "preprocessing.pt"
    if not path.exists():
        if any(model_dir.glob("msa_*.pt")):
            raise RuntimeError(
                "Model checkpoints exist without preprocessing.pt; use a fresh --root"
            )
        return
    saved = torch.load(path, map_location="cpu", weights_only=True)
    old = saved.get("architecture")
    if old is None:
        old = dict(
            name="legacy",
            layers=1,
            hidden=saved["args"]["hidden"],
            dropout=0.0,
            edge_dim=4,
        )
    new = architecture_config(args)
    if old != new:
        raise RuntimeError(
            f"Existing run uses {old}; requested {new}. "
            "Use a fresh --root for this architecture; existing models and results are preserved."
        )
    if "model_code_sha256" not in saved:
        raise RuntimeError(
            "This is the historical completed run. Its checkpoints remain available for "
            "inference/postprocessing; use a fresh --root for training with the revised code."
        )
    if saved["model_code_sha256"] != model_code_hashes():
        raise RuntimeError(
            "Model source changed since this run was trained; use a fresh --root"
        )


def model_code_hashes():
    return {
        name: sha256_file(Path(__file__).with_name(name))
        for name in ("train_cross_msa.py",)
    }


def encode_nodes(enc, x, pack, zero_shot=False):
    """Encode nodes using the training message edges."""
    if zero_shot:
        src = torch.empty(0, dtype=torch.long, device=x.device)
        tgt = torch.empty(0, dtype=torch.long, device=x.device)
        weights = x.new_empty(0)
    else:
        src, tgt, weights = pack["src_mp_t"], pack["tgt_mp_t"], pack["w_mp_t"]
    return enc(x, src, tgt, weights, pack["n_local"])


class BiSAGELayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.W_self = nn.Linear(in_dim, out_dim, bias=False)
        self.W_neigh = nn.Linear(in_dim, out_dim, bias=False)
        self.b = nn.Parameter(torch.zeros(out_dim))

    def forward(self, h, src, tgt, edge_w, n):
        msg_st = h[src] * edge_w.unsqueeze(-1)
        msg_ts = h[tgt] * edge_w.unsqueeze(-1)
        agg = torch.zeros(n, h.shape[1], device=h.device, dtype=h.dtype)
        agg.index_add_(0, tgt, msg_st)
        agg.index_add_(0, src, msg_ts)
        return torch.relu(self.W_self(h) + self.W_neigh(agg) + self.b)


class BiSAGE(nn.Module):
    def __init__(self, in_dim, hidden, layers=1):
        super().__init__()
        dims = [in_dim] + [hidden] * layers
        self.layers = nn.ModuleList(
            [BiSAGELayer(dims[i], dims[i + 1]) for i in range(layers)]
        )

    def forward(self, x, src, tgt, edge_w, n):
        h = x
        for layer in self.layers:
            h = layer(h, src, tgt, edge_w, n)
        return h


class SymEdgeHead(nn.Module):
    def __init__(self, h_dim, edge_dim, hidden=128, out_dim=1):
        super().__init__()
        in_dim = 3 * h_dim + edge_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, h_src, h_tgt, e):
        x = torch.cat([h_src + h_tgt, h_src * h_tgt, (h_src - h_tgt).abs(), e], dim=-1)
        out = self.net(x)
        return out.squeeze(-1) if out.shape[-1] == 1 else out


class BGEProjector(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.norm = nn.LayerNorm(in_dim)
        self.proj = nn.Linear(in_dim, out_dim)
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.proj(self.norm(x)))


def load_all(args):
    print("[load] nodes + edges + meta", flush=True)
    meta = json.loads((BUILD / "meta.json").read_text())
    if meta.get("split_strategy") != "per_msa_unordered_pair":
        raise ValueError("Run 01_build_network.py to regenerate the unordered-pair split")
    acs_cols = list(meta["acs_cols"])
    columns = ["log_area", "ni_out", "ni_in", "top_cat_id", "msa_geoid"] + acs_cols
    if not args.no_name_emb:
        columns.append("name_emb")
    if not args.no_cat_emb:
        columns.append("category_emb")
    nodes = pd.read_parquet(BUILD / "nodes.parquet", columns=columns)
    edges = pd.read_parquet(BUILD / "edges.parquet")

    pop_raw = nodes["ni_out"].to_numpy().astype(np.float64)

    preprocessing = {}
    nodes["ni_out"] = np.log1p(nodes["ni_out"].astype(np.float32))
    nodes["ni_in"] = np.log1p(nodes["ni_in"].astype(np.float32))
    for col in ("ni_out", "ni_in", "log_area"):
        m = nodes[col].mean()
        s = nodes[col].std() or 1.0
        preprocessing[col] = {"mean": float(m), "std": float(s)}
        nodes[col] = (nodes[col] - m) / s

    dense_cols = ["log_area", "ni_out", "ni_in"] + acs_cols
    X_dense = nodes[dense_cols].to_numpy().astype(np.float32)
    X_name = (
        None
        if args.no_name_emb
        else np.stack(nodes["name_emb"].to_numpy()).astype(np.float32)
    )
    X_cat = (
        None
        if args.no_cat_emb
        else np.stack(nodes["category_emb"].to_numpy()).astype(np.float32)
    )
    cat_id = nodes["top_cat_id"].to_numpy().astype(np.int64)
    K_cat = int(cat_id.max() + 1)
    bge_dim = int(meta["bge_dim"])

    log_dist_mean = float(edges["log_distance_km"].mean())
    log_dist_std = float(edges["log_distance_km"].std() or 1.0)
    edges["log_distance_km_std"] = (
        (edges["log_distance_km"] - log_dist_mean) / log_dist_std
    ).astype(np.float32)
    preprocessing["log_distance_km"] = {"mean": log_dist_mean, "std": log_dist_std}
    log_dist_raw = edges["log_distance_km"].to_numpy().astype(np.float64)

    print("[load] MSA membership from nodes.parquet (msa_geoid)", flush=True)
    node_msa = nodes["msa_geoid"].fillna("??").astype(str).to_numpy()
    print(
        f"    nodes with an MSA: {int((node_msa != '??').sum()):,}/{len(nodes):,}",
        flush=True,
    )

    src = edges["src_idx"].to_numpy()
    tgt = edges["tgt_idx"].to_numpy()

    if "msa_geoid" in edges.columns:
        edge_msa = edges["msa_geoid"].astype(str).to_numpy()
    else:
        edge_msa = node_msa[src]

    Xe = (
        edges[["log_distance_km_std", "same_cbg", "same_top_cat", "same_county"]]
        .to_numpy()
        .astype(np.float32)
    )

    if not np.isfinite(X_dense).all() or not np.isfinite(Xe).all():
        raise ValueError("Nonfinite node or edge features")
    if not np.isfinite(edges["log1p_nij"]).all() or (edges["nij"] < 0).any():
        raise ValueError("Nonfinite/negative targets")
    if not set(edges["split"].unique()) <= {"train", "val", "test"}:
        raise ValueError("Unknown split labels")
    if not np.array_equal(node_msa[src], node_msa[tgt]) or not np.array_equal(
        node_msa[src], edge_msa
    ):
        raise ValueError("Training candidates must be within one MSA")
    return dict(
        X_dense=X_dense,
        X_name=X_name,
        X_cat=X_cat,
        cat_id=cat_id,
        K_cat=K_cat,
        bge_dim=bge_dim,
        src=src,
        tgt=tgt,
        Xe=Xe,
        y=edges["log1p_nij"].to_numpy().astype(np.float32),
        raw=edges["nij"].to_numpy().astype(np.float64),
        split=edges["split"].to_numpy(),
        neg_kind=edges["neg_kind"].to_numpy(),
        edge_msa=edge_msa,
        node_msa=node_msa,
        pop_raw=pop_raw,
        log_dist_raw=log_dist_raw,
        log_pop=np.log1p(pop_raw),
        node_groups=pd.Series(node_msa).groupby(node_msa, sort=False).indices,
        edge_groups=pd.Series(edge_msa).groupby(edge_msa, sort=False).indices,
        n_nodes=len(nodes),
        preprocessing=preprocessing,
        dense_cols=dense_cols,
        build_meta=meta,
    )


def fit_gravity(pop_raw, log_dist_raw_rows, src_rows, tgt_rows, y_rows, train_mask):
    log_pop = np.log1p(pop_raw)
    feats = np.stack(
        [log_pop[src_rows], log_pop[tgt_rows], log_dist_raw_rows], axis=1
    ).astype(np.float64)
    lr = LinearRegression()
    lr.fit(feats[train_mask], y_rows[train_mask].astype(np.float64))
    coef = lr.coef_.copy()
    avg = 0.5 * (coef[0] + coef[1])
    base = (lr.intercept_ + feats @ np.array([avg, avg, coef[2]])).astype(np.float32)
    return base, dict(
        intercept=float(lr.intercept_), b_pop=float(avg), b_dist=float(coef[2])
    )


def build_msa_pack(m, D, device, args):
    """Gather one MSA's local node subgraph + its intra-MSA edges (on GPU).
    Node-feature blocks disabled via ablation flags are left as None and never
    moved to GPU (saves memory)."""
    nodes_m = D["node_groups"][m].astype(np.int64)
    n_local = len(nodes_m)
    if n_local < 2:
        return None
    rows = D["edge_groups"][m].astype(np.int64)
    if len(rows) == 0:
        return None
    g2l = -np.ones(D["n_nodes"], dtype=np.int64)
    g2l[nodes_m] = np.arange(n_local)
    s_loc = g2l[D["src"][rows]]
    t_loc = g2l[D["tgt"][rows]]
    sp = D["split"][rows]
    raw_r = D["raw"][rows]
    y_r = D["y"][rows]

    tp = (sp == "train") & (raw_r > 0)
    smp = s_loc[tp]
    tmp = t_loc[tp]
    wraw = y_r[tp]
    if len(smp) > 0:
        deg = np.zeros(n_local)
        np.add.at(deg, smp, wraw)
        np.add.at(deg, tmp, wraw)
        wmp = (wraw / (np.sqrt(np.maximum(deg[smp] * deg[tmp], 1e-9)) + 1e-9)).astype(
            np.float32
        )
        src_mp_t = torch.from_numpy(smp).to(device)
        tgt_mp_t = torch.from_numpy(tmp).to(device)
        w_mp_t = torch.from_numpy(wmp).to(device)
    else:
        src_mp_t = torch.empty(0, dtype=torch.long, device=device)
        tgt_mp_t = torch.empty(0, dtype=torch.long, device=device)
        w_mp_t = torch.empty(0, dtype=torch.float32, device=device)

    return dict(
        m=m,
        n_local=n_local,
        rows=rows,
        sp=sp,
        raw=raw_r,
        src_g=D["src"][rows],
        tgt_g=D["tgt"][rows],
        log_dist_rows=D["log_dist_raw"][rows],
        Xd=torch.from_numpy(D["X_dense"][nodes_m]).to(device),
        Xn=(
            None
            if args.no_name_emb
            else torch.from_numpy(D["X_name"][nodes_m]).to(device)
        ),
        Xc=(
            None
            if args.no_cat_emb
            else torch.from_numpy(D["X_cat"][nodes_m]).to(device)
        ),
        cid=(
            None
            if args.no_cat_id
            else torch.from_numpy(D["cat_id"][nodes_m]).to(device)
        ),
        s_loc=torch.from_numpy(s_loc).to(device),
        t_loc=torch.from_numpy(t_loc).to(device),
        Xe=torch.from_numpy(D["Xe"][rows]).to(device),
        y=torch.from_numpy(y_r.astype(np.float32)).to(device),
        src_mp_t=src_mp_t,
        tgt_mp_t=tgt_mp_t,
        w_mp_t=w_mp_t,
        edge_attr_mp_t=torch.from_numpy(D["Xe"][rows][tp].astype(np.float32)).to(
            device
        ),
    )


def assemble_node_input(pack, proj_n, proj_c, cat_em):
    """Concatenate the enabled node-feature blocks: dense numeric (always) +
    optional name BGE + optional category BGE + optional category-ID embedding.
    A disabled block is passed in as a None module and simply skipped."""
    parts = [pack["Xd"]]
    if proj_n is not None:
        parts.append(proj_n(pack["Xn"]))
    if proj_c is not None:
        parts.append(proj_c(pack["Xc"]))
    if cat_em is not None:
        parts.append(cat_em(pack["cid"]))
    return torch.cat(parts, dim=1)


def train_one(pack, D, args, device, name=""):
    m = pack["m"]
    print(f"\n========== train on MSA {m} ({name}) ==========", flush=True)
    sp = pack["sp"]
    raw_r = pack["raw"]
    train_mask = sp == "train"
    val_mask = sp == "val"
    print(
        f"    nodes={pack['n_local']:,}  edges={len(sp):,}  "
        f"train_pos={int((train_mask & (raw_r>0)).sum()):,}",
        flush=True,
    )

    base_all, base_coefs = fit_gravity(
        D["pop_raw"],
        pack["log_dist_rows"],
        pack["src_g"],
        pack["tgt_g"],
        D["y"][pack["rows"]],
        train_mask,
    )
    yres_t = torch.from_numpy((D["y"][pack["rows"]] - base_all).astype(np.float32)).to(
        device
    )

    bin_edges = np.array(
        [0, 11, 15, 20, 30, 50, 100, 200, 500, 2000, np.inf], dtype=np.float64
    )
    bin_id = np.clip(np.digitize(raw_r, bin_edges) - 1, 0, len(bin_edges) - 2).astype(
        np.int64
    )
    counts = np.bincount(bin_id[train_mask], minlength=len(bin_edges) - 1)
    bin_w = (
        (1.0 / np.maximum(counts.astype(np.float64), 1.0)) ** args.weight_pow
    ).astype(np.float32)
    bin_w = bin_w / bin_w.mean()
    sw_t = torch.from_numpy(bin_w[bin_id].astype(np.float32)).to(device)

    train_idx = torch.from_numpy(np.where(train_mask)[0]).to(device)
    val_idx = torch.from_numpy(np.where(val_mask)[0]).to(device)

    proj_n = (
        None
        if args.no_name_emb
        else BGEProjector(D["bge_dim"], args.bge_proj_dim).to(device)
    )
    proj_c = (
        None
        if args.no_cat_emb
        else BGEProjector(D["bge_dim"], args.bge_proj_dim).to(device)
    )
    cat_em = (
        None
        if args.no_cat_id
        else nn.Embedding(D["K_cat"], args.cat_emb_dim).to(device)
    )
    in_dim = (
        pack["Xd"].shape[1]
        + (0 if proj_n is None else args.bge_proj_dim)
        + (0 if proj_c is None else args.bge_proj_dim)
        + (0 if cat_em is None else args.cat_emb_dim)
    )
    enc = make_encoder(architecture_config(args), in_dim).to(device)
    head = SymEdgeHead(args.hidden, pack["Xe"].shape[1], hidden=args.hidden).to(device)
    params = list(enc.parameters()) + list(head.parameters())
    for mod in (cat_em, proj_n, proj_c):
        if mod is not None:
            params += list(mod.parameters())
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-5)

    s_loc, t_loc, Xe_t = pack["s_loc"], pack["t_loc"], pack["Xe"]

    def make_input():
        return assemble_node_input(pack, proj_n, proj_c, cat_em)

    def eval_loss(idx):
        with torch.no_grad():
            h = encode_nodes(enc, make_input(), pack)
            losses = []
            for s_ in range(0, idx.numel(), args.batch):
                sub = idx[s_ : s_ + args.batch]
                pred = head(h[s_loc[sub]], h[t_loc[sub]], Xe_t[sub])
                losses.append(((pred - yres_t[sub]) ** 2).detach())
        return float(torch.cat(losses).mean()) if losses else float("nan")

    best_val = float("inf")
    bad = 0
    best_state = None
    best_epoch = 0
    history = []
    stop_reason = "epoch_limit"
    t0 = time.time()
    for ep in range(1, args.epochs + 1):
        if args.warmup_epochs > 0 and ep <= args.warmup_epochs:
            for g in opt.param_groups:
                g["lr"] = args.lr * ep / max(1, args.warmup_epochs)
        elif ep == args.warmup_epochs + 1:
            for g in opt.param_groups:
                g["lr"] = args.lr
        if train_idx.numel() == 0:
            stop_reason = "no_training_edges"
            print("    no training edges; skipping", flush=True)
            break
        enc.train()
        head.train()
        perm = torch.randperm(train_idx.numel(), device=device)
        tot = 0.0
        nb = 0
        for s_ in range(0, train_idx.numel(), args.batch):
            idx = train_idx[perm[s_ : s_ + args.batch]]
            h = encode_nodes(enc, make_input(), pack)
            pred = head(h[s_loc[idx]], h[t_loc[idx]], Xe_t[idx])
            loss = (sw_t[idx] * (pred - yres_t[idx]) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item()
            nb += 1
        enc.eval()
        head.eval()
        va = eval_loss(val_idx) if val_idx.numel() > 0 else float("nan")
        if np.isfinite(va) and va < best_val - 1e-5:
            best_val = va
            bad = 0
            best_epoch = ep
            best_state = {
                n_: {k: v.detach().cpu().clone() for k, v in mod.state_dict().items()}
                for n_, mod in [
                    ("enc", enc),
                    ("head", head),
                    ("cat", cat_em),
                    ("proj_n", proj_n),
                    ("proj_c", proj_c),
                ]
                if mod is not None
            }
        else:
            bad += 1
        history.append(
            dict(
                epoch=ep,
                train_weighted_mse=float(tot / max(nb, 1)),
                validation_mse=float(va),
                best_val=float(best_val),
                learning_rate=float(opt.param_groups[0]["lr"]),
                epochs_without_improvement=int(bad),
            )
        )
        if ep <= 3 or ep % 10 == 0 or bad > args.patience:
            print(
                f"    ep {ep:02d}  train_mse={tot/max(nb,1):.4f}  val_mse={va:.4f}  "
                f"best={best_val:.4f}  bad={bad}",
                flush=True,
            )
        if bad > args.patience:
            stop_reason = "early_stopping"
            print(f"    early stop @ ep{ep}", flush=True)
            break
    print(f"    done in {time.time()-t0:.1f}s, best_val={best_val:.4f}", flush=True)

    if best_state is not None:
        enc.load_state_dict({k: v.to(device) for k, v in best_state["enc"].items()})
        head.load_state_dict({k: v.to(device) for k, v in best_state["head"].items()})
        if cat_em is not None:
            cat_em.load_state_dict(
                {k: v.to(device) for k, v in best_state["cat"].items()}
            )
        if proj_n is not None:
            proj_n.load_state_dict(
                {k: v.to(device) for k, v in best_state["proj_n"].items()}
            )
        if proj_c is not None:
            proj_c.load_state_dict(
                {k: v.to(device) for k, v in best_state["proj_c"].items()}
            )
    return dict(
        enc=enc,
        head=head,
        cat_em=cat_em,
        proj_n=proj_n,
        proj_c=proj_c,
        base_coefs=base_coefs,
        best_val=best_val,
        best_epoch=best_epoch,
        epochs_run=ep,
        train_seconds=time.time() - t0,
        max_epochs=int(args.epochs),
        stop_reason=stop_reason,
        epochs_without_improvement=int(bad),
        history=history,
    )


def eval_cross(model_pack, pack_test, D, args, device, zero_shot=False):
    sp = pack_test["sp"]
    test_mask = sp == "test"
    if int(test_mask.sum()) < 50:
        return dict(
            n=int(test_mask.sum()), pearson_full=np.nan, r2_full=np.nan, mae_full=np.nan
        )

    enc = model_pack["enc"]
    head = model_pack["head"]
    cat_em = model_pack["cat_em"]
    proj_n = model_pack["proj_n"]
    proj_c = model_pack["proj_c"]
    enc.eval()
    head.eval()
    for mod in (proj_n, proj_c):
        if mod is not None:
            mod.eval()

    with torch.no_grad():
        x = assemble_node_input(pack_test, proj_n, proj_c, cat_em)
        h = encode_nodes(enc, x, pack_test, zero_shot=zero_shot)
        idx_np = np.where(test_mask)[0]
        idx = torch.from_numpy(idx_np).to(device)
        s_loc, t_loc, Xe_t = pack_test["s_loc"], pack_test["t_loc"], pack_test["Xe"]
        preds = []
        for s_ in range(0, idx.numel(), args.batch):
            sub = idx[s_ : s_ + args.batch]
            preds.append(head(h[s_loc[sub]], h[t_loc[sub]], Xe_t[sub]).cpu().numpy())
        res_pred = np.concatenate(preds)

    bc = model_pack["base_coefs"]
    log_pop = D["log_pop"]
    sg = pack_test["src_g"][idx_np]
    tg = pack_test["tgt_g"][idx_np]
    base = (
        bc["intercept"]
        + bc["b_pop"] * log_pop[sg]
        + bc["b_pop"] * log_pop[tg]
        + bc["b_dist"] * pack_test["log_dist_rows"][idx_np]
    ).astype(np.float32)
    pred_log = np.maximum(base + res_pred, 0.0)

    y_true = D["y"][pack_test["rows"][idx_np]]

    def _pe(a, b):
        return (
            float(np.corrcoef(a, b)[0, 1])
            if len(a) > 1 and a.std() > 0 and b.std() > 0
            else np.nan
        )

    def _metrics(y, pred):
        y, pred = y.astype(np.float64), pred.astype(np.float64)
        variance = np.square(y - y.mean()).sum() if len(y) else 0.0
        return (
            float(1 - np.square(y - pred).sum() / variance) if variance else np.nan,
            float(np.abs(y - pred).mean()) if len(y) else np.nan,
        )

    r2_full, mae_full = _metrics(y_true, pred_log)
    return dict(
        n=int(test_mask.sum()),
        pearson_full=_pe(y_true, pred_log),
        r2_full=r2_full,
        mae_full=mae_full,
    )


def load_checkpoint(path, pack, D, args, device, signature=None):
    state = torch.load(path, map_location="cpu", weights_only=True)
    if signature is not None and state["signature"] != signature:
        raise RuntimeError(f"Checkpoint configuration mismatch: {path}")
    config = checkpoint_architecture(state)
    model = dict(
        enc=make_encoder(config, pack["Xd"].shape[1]),
        head=SymEdgeHead(
            config["hidden"], pack["Xe"].shape[1], hidden=config["hidden"]
        ),
        proj_n=None,
        proj_c=None,
        cat_em=None,
    )
    for key, module in model.items():
        if module is not None:
            module.load_state_dict(state["modules"][key])
            module.to(device).eval()
    model.update(state["training"])
    model["architecture"] = config
    return model


def main():
    args = parse_args()
    configure_paths(args.root)
    check_existing_architecture(ROOT, args)
    artifacts, model_dir = ROOT / ".work", ROOT / "models"
    work = artifacts / "training"
    row_dir = work / "rows"
    for directory in (artifacts, model_dir, work, row_dir):
        directory.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(args.threads)
    device = args.device
    if not str(device).startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError(
            "The full 376-by-376 run requires the accessible CUDA GPU. Run on the GPU-visible host."
        )
    torch.cuda.set_device(torch.device(device))
    print(
        f"GPU: {torch.cuda.get_device_name()} | torch={torch.__version__} | CUDA={torch.version.cuda}",
        flush=True,
    )
    D = load_all(args)
    source_counts = pd.Series(D["edge_msa"][D["raw"] > 0]).value_counts()

    ranked = pd.DataFrame(
        {
            "GEOID": source_counts.index.astype(str),
            "n_positive_edges": source_counts.to_numpy(),
        }
    )
    ranked = ranked.sort_values(["n_positive_edges", "GEOID"], ascending=[False, True])
    TOP = ranked.GEOID.head(args.top_msas).tolist()
    summary = pd.read_csv(MSA_SUMMARY, dtype={"GEOID": str})
    names = dict(zip(summary.GEOID, summary.NAME))

    def nm(g):
        return short_msa_name(names.get(str(g), str(g)))

    metadata = []
    for g in TOP:
        rows = D["edge_groups"][g]
        metadata.append(
            dict(
                GEOID=g,
                NAME=names.get(g, g),
                short_name=nm(g),
                n_nodes=len(D["node_groups"][g]),
                n_positive_edges=int(source_counts[g]),
                n_candidates=len(rows),
                n_train=int((D["split"][rows] == "train").sum()),
                n_val=int((D["split"][rows] == "val").sum()),
                n_test=int((D["split"][rows] == "test").sum()),
            )
        )
    print(
        f'Validated {len(TOP)} MSAs, {D["n_nodes"]:,} nodes, {len(D["raw"]):,} candidates',
        flush=True,
    )
    print("Preparing local MSA tensors", flush=True)
    packs = {g: build_msa_pack(g, D, device, args) for g in TOP}
    assert all(p is not None for p in packs.values())
    N = len(TOP)
    manifest_args = {
        k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
    }
    manifest = dict(
        args=manifest_args,
        top_msas=TOP,
        build_meta=D["build_meta"],
        preprocessing=D["preprocessing"],
        dense_cols=D["dense_cols"],
        torch_version=str(torch.__version__),
        cuda_version=torch.version.cuda,
        gpu=torch.cuda.get_device_name(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        protocol="Source TRAIN-only model fitting; target TRAIN-positive graph available; no target fine-tuning",
        limitations="Directed-row split allows reverse-edge overlap; all-positive NI aggregates and global feature normalization retained; one seeded run",
    )
    manifest["architecture"] = architecture_config(args)
    manifest["model_code_sha256"] = model_code_hashes()

    manifest["build_table_sha256"] = {
        name: sha256_file(BUILD / name) for name in ("nodes.parquet", "edges.parquet")
    }
    signature = resolve_run_signature(manifest, model_dir)
    manifest_path = work / "run_manifest.json"
    if (
        manifest_path.exists()
        and json.loads(manifest_path.read_text())["signature"] != signature
    ):
        raise RuntimeError(
            "Data/configuration/code changed: use a fresh experiment folder or restore the matching inputs"
        )
    pd.DataFrame(metadata).sort_values("GEOID").to_csv(
        artifacts / "msa_metadata.csv", index=False
    )
    manifest_path.write_text(
        json.dumps(dict(signature=signature, **manifest), indent=2) + "\n"
    )
    torch.save(
        dict(
            signature=signature,
            preprocessing=D["preprocessing"],
            dense_cols=D["dense_cols"],
            args=manifest_args,
            build_meta=D["build_meta"],
            architecture=manifest["architecture"],
            model_code_sha256=manifest["model_code_sha256"],
            build_table_sha256=manifest["build_table_sha256"],
            edge_features=[
                "log_distance_km_std",
                "same_cbg",
                "same_top_cat",
                "same_county",
            ],
            node_transform="log1p(ni_out/ni_in), then global mean/sample-SD; log_area global mean/sample-SD; ACS builder scaling",
            protocol=manifest["protocol"],
            limitations=manifest["limitations"],
        ),
        model_dir / "preprocessing.pt",
    )
    rows = []
    training_records = []
    started = time.time()

    def progress(stage, source=None, completed=0):
        data = dict(
            stage=stage,
            source_msa=source,
            completed_sources=completed,
            total_sources=N,
            completed_cells=len(rows),
            device=device,
            signature=signature,
            updated_unix=time.time(),
        )
        tmp = work / "progress.json.tmp"
        tmp.write_text(json.dumps(data, indent=2) + "\n")
        tmp.replace(work / "progress.json")

    for i, tr in enumerate(TOP):
        row_path = row_dir / f"{tr}.csv"
        checkpoint = model_dir / f"msa_{tr}.pt"
        if row_path.exists():
            saved = pd.read_csv(row_path, dtype={"train_geoid": str, "test_geoid": str})
            if (
                len(saved) != N
                or saved.test_geoid.tolist() != TOP
                or not saved.train_geoid.eq(tr).all()
            ):
                raise RuntimeError(f"Incomplete or mismatched saved row: {row_path}")
            if not checkpoint.exists():
                raise RuntimeError(
                    f"Resume row exists without its final model: {checkpoint}"
                )
            state = torch.load(checkpoint, map_location="cpu", weights_only=True)
            if state["signature"] != signature:
                raise RuntimeError(f"Checkpoint signature mismatch: {checkpoint}")
            training = state["training"]
            row_results = saved.to_dict("records")
            print(f"[resume] completed source {tr}", flush=True)

        else:
            seed = args.seed + int(tr)
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            progress("training", tr, i)
            if checkpoint.exists():
                model = load_checkpoint(
                    checkpoint, packs[tr], D, args, device, signature
                )
            else:
                model = train_one(packs[tr], D, args, device, name=nm(tr))
                modules = {
                    key: {
                        k: value.detach().cpu()
                        for k, value in model[key].state_dict().items()
                    }
                    for key in ("enc", "head")
                }
                training = {key: model[key] for key in TRAINING_RECORD_KEYS}
                tmp = checkpoint.with_suffix(".tmp")
                torch.save(
                    dict(
                        signature=signature,
                        source_geoid=tr,
                        source_name=names.get(tr, tr),
                        seed=seed,
                        modules=modules,
                        training=training,
                        preprocessing=D["preprocessing"],
                        dense_cols=D["dense_cols"],
                        hidden=args.hidden,
                        edge_dim=4,
                        architecture=manifest["architecture"],
                    ),
                    tmp,
                )
                tmp.replace(checkpoint)
            training = {key: model[key] for key in TRAINING_RECORD_KEYS}
            progress("evaluating", tr, i)
            row_results = []
            for te in TOP:
                metrics = eval_cross(model, packs[te], D, args, device)
                row_results.append(
                    dict(
                        train_geoid=tr,
                        train_name=nm(tr),
                        test_geoid=te,
                        test_name=nm(te),
                        **metrics,
                    )
                )
            tmp = row_path.with_suffix(".tmp")
            pd.DataFrame(row_results).to_csv(tmp, index=False)
            tmp.replace(row_path)
            del model
        rows.extend(row_results)
        training_records.append(
            dict(
                GEOID=tr,
                NAME=names.get(tr, tr),
                seed=args.seed + int(tr),
                best_val=training["best_val"],
                best_epoch=training["best_epoch"],
                epochs_run=training["epochs_run"],
                max_epochs=training["max_epochs"],
                stop_reason=training["stop_reason"],
                epochs_without_improvement=training["epochs_without_improvement"],
                train_seconds=training["train_seconds"],
            )
        )
        progress("source_complete", tr, i + 1)
        diag = next(r["pearson_full"] for r in row_results if r["test_geoid"] == tr)
        print(
            f"ROW {i + 1}/{N}: {tr} ({nm(tr)}) within r={diag:+.4f}; elapsed {time.time() - started:.1f}s",
            flush=True,
        )
    metrics = pd.DataFrame(rows)
    assert (
        len(metrics) == N * N
        and not metrics.duplicated(["train_geoid", "test_geoid"]).any()
    )
    metric_cols = ["pearson_full", "r2_full", "mae_full"]
    if not np.isfinite(metrics[metric_cols].to_numpy()).all():
        raise ValueError("Nonfinite transfer metrics in completed source rows")
    matrix = metrics.pivot(
        index="train_geoid", columns="test_geoid", values="pearson_full"
    ).reindex(index=sorted(TOP), columns=sorted(TOP))
    matrix.index.name = "source_msa"
    matrix_dir = artifacts / "gnn_all_pairs"
    matrix_dir.mkdir(exist_ok=True)
    matrix.to_csv(matrix_dir / "transfer_pearson.csv")
    within = metrics.loc[metrics.train_geoid.eq(metrics.test_geoid)].copy()
    within = within.rename(
        columns={"train_geoid": "GEOID", "train_name": "short_name"}
    ).drop(columns=["test_geoid", "test_name"])
    within = within.merge(
        pd.DataFrame(training_records), on="GEOID", validate="one_to_one"
    )

    within["improved_after_first_epoch"] = within.best_epoch.gt(1)
    within["retained_for_clustering"] = True
    within["retention_reason"] = "all_trained_msas_included"
    within["validation_min_delta"] = 1e-5
    within = within.rename(
        columns={
            "GEOID": "geoid",
            "NAME": "name",
            "pearson_full": "gnn_r",
            "n": "n_test",
        }
    )
    within.sort_values("geoid").to_csv(
        artifacts / "within_msa_performance.csv", index=False
    )
    settings = {
        **manifest_args,
        "n_msas": N,
        "n_transfer_cells": N * N,
        "seed_rule": "seed + integer source GEOID",
        "split_strategy": D["build_meta"]["split_strategy"],
        "message_graph": "target TRAIN-positive graph",
        "label": "log1p(NIJ)",
        "prediction": "max(0, source gravity prediction + GNN residual)",
        "run_signature": signature,
        "seconds": time.time() - started,
        "script_sha256": manifest["script_sha256"],
        "executed_script_sha256": manifest["executed_script_sha256"],
        "training_protocol_sha256": manifest["training_protocol_sha256"],
        "message_edge_features": "none",
        "message_edge_weight": "Normalized log1p(NIJ)",
        "torch_version": str(torch.__version__),
        "device_name": torch.cuda.get_device_name(),
    }
    pd.DataFrame([{"setting": k, "value": v} for k, v in settings.items()]).to_csv(
        artifacts / "training_settings.csv", index=False
    )
    progress("complete", completed=N)
    print(
        f"COMPLETE: {N} fitted models; {N*N:,} source-target cells; one GNN transfer matrix.",
        flush=True,
    )


if __name__ == "__main__":
    main()
