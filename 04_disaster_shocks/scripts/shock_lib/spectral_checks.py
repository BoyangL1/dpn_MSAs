"""Assess a nonnegative Leontief matrix without modifying or rescaling it.

An eigenvalue estimate alone does not determine the reported classification.
For every nonzero x >= 0, L = min_{x_i > 0} (W x)_i / x_i is a lower
bound on rho(W), because W x >= L x also holds on the zero coordinates.
For x > 0 the corresponding maximum is an upper bound. The maximum row
sum supplies an upper bound even when an eigenvector has zero entries.
These are floating-point bounds with explicit conservative guards, not
interval-arithmetic certificates. Cases within the decision tolerance of
the critical point remain unresolved. No matrix normalization is applied.
"""

from __future__ import annotations
import numpy as np
import scipy.sparse as sp
import scipy.sparse.csgraph as csgraph
import scipy.sparse.linalg as sla

DECISION_TOL = 1e-09


def assess_criticality(W, beta):
    """Return spectral estimates, guarded bounds and a conservative status.

    ``subcritical`` is True only when the guarded upper bound on beta*rho
    is below 1 - DECISION_TOL. Supercriticality requires a lower bound above
    1 + DECISION_TOL. ``beta_critical`` is infinity for an exactly zero
    spectral radius. Inputs are never changed; beta must be nonnegative.
    """
    beta = float(beta)
    if not np.isfinite(beta) or beta < 0:
        raise ValueError("beta must be finite and nonnegative")
    matrix = sp.csr_matrix(W, copy=True)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("W must be square")
    if np.iscomplexobj(matrix.data):
        if np.any(matrix.data.imag != 0):
            raise ValueError("W must be real and nonnegative")
        matrix.data = matrix.data.real
    matrix = matrix.astype(np.float64)
    if not np.isfinite(matrix.data).all() or np.any(matrix.data < 0):
        raise ValueError("W must have finite nonnegative entries")
    matrix.sum_duplicates()
    matrix.eliminate_zeros()
    matrix.sort_indices()
    if not np.isfinite(matrix.data).all():
        raise ValueError("Summed matrix entries overflowed")
    n = matrix.shape[0]
    row_sums = np.asarray(matrix.sum(axis=1)).ravel()
    if not np.isfinite(row_sums).all():
        raise ValueError("Matrix row sums overflowed")
    row_max = float(row_sums.max(initial=0.0))
    max_degree = int(np.diff(matrix.indptr).max(initial=0))
    relative_guard = max(1e-12, 16 * np.finfo(float).eps * (max_degree + 3))

    def guard(value):
        return relative_guard * max(1.0, abs(float(value)))

    lower = max(0.0, float(matrix.diagonal().max(initial=0.0)))
    lower = max(0.0, lower - guard(lower))
    upper = row_max + guard(row_max) if row_max else 0.0
    lower_method, upper_method = ("diagonal", "maximum_row_sum")
    lower_support, lower_cutoff = (0, None)

    def classify():
        if beta * upper < 1 - DECISION_TOL:
            return "subcritical"
        if beta * lower > 1 + DECISION_TOL:
            return "supercritical"
        return "unresolved"

    def consider(vector, method, cutoff=None):
        nonlocal lower, upper, lower_method, upper_method, lower_support, lower_cutoff
        x = np.asarray(vector, dtype=float).ravel().copy()
        if x.shape != (n,) or not np.isfinite(x).all() or np.any(x < 0):
            return
        scale = float(x.max(initial=0.0))
        if scale <= 0:
            return
        x /= scale
        if cutoff is not None:
            x[x < cutoff] = 0.0
        support = x > 0
        if not support.any():
            return
        product = matrix @ x
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            ratios = product[support] / x[support]
        raw_lower = float(ratios.min())
        if np.isfinite(raw_lower):
            candidate = max(0.0, raw_lower - guard(raw_lower))
            if candidate > lower:
                lower = candidate
                lower_method = method
                lower_support = int(support.sum())
                lower_cutoff = cutoff
        if support.all():
            raw_upper = float(ratios.max())
            if np.isfinite(raw_upper):
                candidate = raw_upper + guard(raw_upper)
                if candidate < upper:
                    upper = candidate
                    upper_method = method

    eigenvalue = 0.0 + 0j
    eigenvector = np.ones(n)
    estimate_method = "zero_matrix"
    converged = True
    error = None
    exact_radius = None
    if not matrix.nnz:
        exact_radius = 0.0
    elif n <= 32:
        values, vectors = np.linalg.eig(matrix.toarray())
        which = int(np.argmax(values.real))
        eigenvalue, eigenvector = (values[which], vectors[:, which])
        estimate_method = "dense_eig_largest_real"
    else:
        n_components = csgraph.connected_components(
            matrix, directed=True, connection="strong", return_labels=False
        )
        if n_components == n:
            exact_radius = float(matrix.diagonal().max(initial=0.0))
            eigenvalue = complex(exact_radius)
            estimate_method = "singleton_scc_exact"
        else:
            try:
                values, vectors = sla.eigs(
                    matrix,
                    k=1,
                    which="LR",
                    v0=np.linspace(1.0, 2.0, n),
                    tol=1e-11,
                    maxiter=min(20000, max(2000, 20 * n)),
                )
                eigenvalue, eigenvector = (values[0], vectors[:, 0])
                estimate_method = "arpack_largest_real"
            except sla.ArpackNoConvergence as exc:
                converged = False
                error = type(exc).__name__
                if exc.eigenvalues is not None and len(exc.eigenvalues):
                    which = int(np.argmax(exc.eigenvalues.real))
                    eigenvalue = exc.eigenvalues[which]
                    eigenvector = exc.eigenvectors[:, which]
                    estimate_method = "arpack_partial_largest_real"
                else:
                    estimate_method = "shifted_power_fallback"
            except (sla.ArpackError, RuntimeError, ValueError) as exc:
                converged = False
                error = type(exc).__name__
                estimate_method = "shifted_power_fallback"
    refinement_steps = 0
    if exact_radius is not None:
        if exact_radius == 0:
            lower = upper = 0.0
        else:
            lower = max(0.0, exact_radius - guard(exact_radius))
            upper = exact_radius + guard(exact_radius)
        lower_method = upper_method = estimate_method
        estimate = exact_radius
        residual = 0.0 if not matrix.nnz else None
    else:
        estimate = float(abs(eigenvalue))
        consider(np.ones(n), "unit_vector")
        magnitude = np.abs(eigenvector)
        for cutoff in (None, 1e-14, 1e-12, 1e-10, 1e-08, 1e-06, 0.0001):
            consider(magnitude, "eigenvector_absolute", cutoff)
        if magnitude.max(initial=0.0) > 0:
            positive = magnitude / magnitude.max()
            for floor in (1e-12, 1e-08, 0.0001):
                consider(positive + floor, f"positive_eigenvector_floor_{floor:g}")
        if classify() == "unresolved" or estimate_method == "shifted_power_fallback":
            x = np.ones(n)
            z = np.ones(n)
            shift = max(estimate, min(row_max, 1.0), 1e-06)
            neumann_active = beta > 0
            for iteration in range(1, 257):
                y = matrix @ x
                shifted = y + shift * x
                norm = float(shifted.max(initial=0.0))
                if norm > 0 and np.isfinite(norm):
                    x = np.maximum(shifted / norm, 1e-250)
                else:
                    break
                if neumann_active:
                    with np.errstate(over="ignore", invalid="ignore"):
                        candidate_z = 1.0 + beta * (matrix @ z)
                    if np.isfinite(candidate_z).all() and candidate_z.max() < 1e100:
                        z = candidate_z
                    else:
                        neumann_active = False
                if iteration % 4 == 0:
                    consider(x, "shifted_power_positive")
                    if neumann_active:
                        consider(z, "positive_neumann_resolvent")
                    if estimate_method == "shifted_power_fallback":
                        eigenvector = x.copy()
                        eigenvalue = complex(float(x @ (matrix @ x) / (x @ x)))
                        estimate = float(abs(eigenvalue))
                    refinement_steps = iteration
                    if classify() != "unresolved":
                        if estimate_method != "shifted_power_fallback":
                            break
                        fallback_error = np.max(
                            np.abs(matrix @ eigenvector - eigenvalue * eigenvector)
                        )
                        fallback_scale = (row_max + abs(eigenvalue)) * np.max(
                            np.abs(eigenvector)
                        )
                        if fallback_error <= 1e-10 * max(
                            fallback_scale, np.finfo(float).tiny
                        ):
                            break
                    if beta * (upper - lower) < DECISION_TOL / 4:
                        break
        if (
            classify() == "unresolved"
            and beta * lower < 1 - DECISION_TOL
            and (beta * upper > 1 + DECISION_TOL)
        ):
            count, labels = csgraph.connected_components(
                matrix, directed=True, connection="strong", return_labels=True
            )
            sizes = np.bincount(labels, minlength=count)
            strength = np.zeros(count)
            np.maximum.at(strength, labels, magnitude)
            order = np.argsort(labels, kind="stable")
            starts = np.r_[0, np.cumsum(sizes)]
            for component in np.argsort(-strength, kind="stable"):
                indices = order[starts[component] : starts[component + 1]]
                if len(indices) < 2:
                    continue
                block = matrix[indices][:, indices].tocsr()
                block_max = float(np.asarray(block.sum(axis=1)).max())
                if beta * (block_max + guard(block_max)) <= 1 + DECISION_TOL:
                    continue
                witness = np.zeros(n)
                witness[indices] = magnitude[indices]
                for cutoff in (None, 1e-14, 1e-12, 1e-10, 1e-08, 1e-06, 0.0001):
                    consider(witness, "scc_restricted_global_eigenvector", cutoff)
                if classify() == "supercritical":
                    break
                try:
                    if len(indices) <= 32:
                        block_values, block_vectors = np.linalg.eig(block.toarray())
                        block_vector = block_vectors[
                            :, int(np.argmax(block_values.real))
                        ]
                    else:
                        _, block_vectors = sla.eigs(
                            block,
                            k=1,
                            which="LR",
                            v0=np.linspace(1.0, 2.0, len(indices)),
                            tol=1e-13,
                            ncv=min(len(indices), 40),
                            maxiter=min(20000, max(2000, 20 * len(indices))),
                        )
                        block_vector = block_vectors[:, 0]
                except (
                    sla.ArpackNoConvergence,
                    sla.ArpackError,
                    RuntimeError,
                    ValueError,
                ):
                    continue
                witness[indices] = np.abs(block_vector)
                for cutoff in (None, 1e-14, 1e-12, 1e-10, 1e-08, 1e-06, 0.0001):
                    consider(witness, "scc_eigenvector_absolute", cutoff)
                if classify() == "supercritical":
                    break
                block_magnitude = np.abs(block_vector)
                peak = float(block_magnitude.max(initial=0.0))
                if peak <= 0:
                    continue
                for support_cutoff in (0.01, 0.001, 0.0001, 1e-06):
                    active = np.flatnonzero(block_magnitude >= support_cutoff * peak)
                    if len(active) < 2 or len(active) == len(indices):
                        continue
                    principal = block[active][:, active].tocsr()
                    try:
                        if len(active) <= 32:
                            values_small, vectors_small = np.linalg.eig(
                                principal.toarray()
                            )
                            small_vector = vectors_small[
                                :, int(np.argmax(values_small.real))
                            ]
                        else:
                            _, vectors_small = sla.eigs(
                                principal,
                                k=1,
                                which="LR",
                                v0=np.linspace(1.0, 2.0, len(active)),
                                tol=1e-13,
                                ncv=min(len(active), 40),
                                maxiter=min(20000, max(2000, 20 * len(active))),
                            )
                            small_vector = vectors_small[:, 0]
                    except (
                        sla.ArpackNoConvergence,
                        sla.ArpackError,
                        RuntimeError,
                        ValueError,
                    ):
                        continue
                    witness.fill(0.0)
                    witness[indices[active]] = np.abs(small_vector)
                    for cutoff in (None, 1e-14, 1e-10, 1e-06):
                        consider(
                            witness,
                            f"scc_principal_submatrix_{support_cutoff:g}",
                            cutoff,
                        )
                    if classify() == "supercritical":
                        break
                if classify() == "supercritical":
                    break
        numerator = float(
            np.max(np.abs(matrix @ eigenvector - eigenvalue * eigenvector), initial=0.0)
        )
        denominator = (row_max + abs(eigenvalue)) * float(
            np.max(np.abs(eigenvector), initial=0.0)
        )
        residual = numerator / denominator if denominator else 0.0
    if lower > upper:
        raise ArithmeticError("Guarded spectral bounds are inconsistent")
    status = classify()
    return dict(
        spectral_radius_raw=float(estimate),
        beta_rho=float(beta * estimate),
        beta_critical=float(1.0 / estimate) if estimate > 0 else float("inf"),
        criticality_status=status,
        subcritical=status == "subcritical",
        rho_lower_bound=float(lower),
        rho_upper_bound=float(upper),
        beta_rho_lower_bound=float(beta * lower),
        beta_rho_upper_bound=float(beta * upper),
        eigen_residual=float(residual) if residual is not None else None,
        estimate_method=estimate_method,
        arpack_converged=bool(converged),
        eigen_solver_error=error,
        rho_lower_bound_method=lower_method,
        rho_upper_bound_method=upper_method,
        lower_bound_support_size=lower_support,
        lower_bound_relative_cutoff=lower_cutoff,
        maximum_row_sum=row_max,
        bound_relative_guard=relative_guard,
        decision_tolerance=DECISION_TOL,
        refinement_steps=refinement_steps,
        eigen_residual_definition="infinity residual / ((maximum row sum + abs(eigenvalue)) * eigenvector infinity norm); null for an exact SCC-spectrum proof without an eigenvector",
        bounds_note="Nonnegative-vector inequalities with floating-point guards; no rescaling or interval arithmetic",
    )
