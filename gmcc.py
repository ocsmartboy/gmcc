"""
Global Mean-Centered Correlation (GMCC)
=======================================

Reference implementation of the similarity measure introduced in:

    Wiyono, S., Adji, T. B., Nugroho, H. A., & Wibirama, S. (2026).
    Global mean-centered correlation: A variance-reduced and theoretically
    grounded similarity measure for sparse data environments.
    Information Sciences, 749, 123502.
    https://doi.org/10.1016/j.ins.2026.123502

Scope
-----
GMCC is a general-purpose similarity measure, not a recommender-system method.
It applies to any dataset of *entities* characterised by partially observed
*attributes* with real-valued measurements -- following Section 3.1 of the
paper, each observed value ``z_k`` in ``R`` is "a measured quantity such as a
rating, score, or feature value".  Ratings appear throughout the examples
in this repository only because they are familiar and easy to read; no part of
the method assumes a rating scale, a bounded range, integer values, or a
recommender-system setting.  The global mean ``mu`` is estimated from whatever
data is supplied, so the measure adapts to the scale of the domain on its own.

GMCC replaces the entity-specific *local* mean centering used by Pearson
Correlation with a single population-level *global* mean ``mu``, computed once
over all observed values in the dataset.  For two value vectors ``x`` and ``y``
defined over ``n`` co-observed attributes (Eq. 6 in the paper)::

                        sum_i (x_i - mu) * (y_i - mu)
    GMCC(x, y) = ---------------------------------------------
                 sqrt(sum_i (x_i - mu)^2) * sqrt(sum_i (y_i - mu)^2)

where the global mean is given by (Eq. 7)::

    mu = (1 / N) * sum_k z_k

over all ``N`` observed values ``z_k`` in the dataset.

The measure is parameter-free: ``mu`` is estimated from data, not tuned.

Properties established in the paper and exercised by ``test_gmcc.py``
---------------------------------------------------------------------
* Bounded in ``[-1, 1]`` and symmetric.
* Equivalent to Pearson Correlation when the centering references are aligned
  (Proposition 2).
* Remains defined for constant-valued profiles, where Pearson is undefined
  (Proposition 3): for ``x = c_x * 1`` and ``y = c_y * 1`` with
  ``c_x != mu`` and ``c_y != mu``, ``GMCC(x, y) = sign((c_x - mu)(c_y - mu))``.
* Baseline variance reduced by a factor ``n / N`` relative to local centering
  (Theorem 1).

Complexity (Section 3.4): ``O(N)`` once for the global mean, then ``O(n)`` per
pair -- identical to Cosine and Pearson -- with only ``O(1)`` extra storage.

Author: Slamet Wiyono
License: MIT
"""

from __future__ import annotations

from typing import Iterable, Optional, Sequence

import numpy as np

__all__ = [
    "global_mean",
    "gmcc",
    "gmcc_from_matrix",
    "gmcc_similarity_matrix",
]

__version__ = "1.0.0"

# Values whose magnitude falls below this are treated as zero when testing
# whether a centered vector has collapsed to the origin.
_EPS = 1e-12


def global_mean(data: Iterable[float] | np.ndarray) -> float:
    """Compute the global mean ``mu`` over all observed values (Eq. 7).

    This is the one-time preprocessing step of GMCC.  Compute it **once** on
    the dataset (for a predictive setting, on the training split only) and
    reuse the returned value for every pairwise similarity computation.

    Parameters
    ----------
    data : array-like
        Observed values of any real-valued quantity -- ratings, scores, sensor
        readings, feature values.  May be any shape; it is flattened before
        averaging.  ``NaN`` entries denote unobserved positions and are
        ignored, so a matrix with missing cells can be passed directly.

    Returns
    -------
    float
        The global mean ``mu``.

    Raises
    ------
    ValueError
        If ``data`` contains no observed (non-``NaN``) value.

    Examples
    --------
    >>> import numpy as np
    >>> R = np.array([[1.0, 1.0, 1.0],
    ...               [5.0, 5.0, 5.0],
    ...               [1.0, 2.0, 3.0],
    ...               [3.0, 4.0, 5.0]])
    >>> float(global_mean(R))
    3.0
    >>> float(global_mean([5.0, np.nan, 1.0]))
    3.0
    """
    arr = np.asarray(data, dtype=float).ravel()
    observed = arr[~np.isnan(arr)]
    if observed.size == 0:
        raise ValueError("No observed values: cannot compute the global mean.")
    return float(observed.mean())


def gmcc(
    x: Sequence[float] | np.ndarray,
    y: Sequence[float] | np.ndarray,
    mu: float,
) -> float:
    """Compute GMCC between two co-observed value vectors (Eq. 6).

    Both vectors must already be restricted to the attributes the two entities
    co-observe, and aligned element-wise.  To derive them from a matrix with
    missing cells, use :func:`gmcc_from_matrix` instead.

    Parameters
    ----------
    x, y : array-like of float
        Value vectors of equal length ``n`` over the co-observed attributes.
    mu : float
        The global mean, as returned by :func:`global_mean`.  It is a property
        of the dataset, not of this pair, and must not be recomputed per pair.

    Returns
    -------
    float
        Similarity in ``[-1, 1]``, or ``nan`` when undefined -- that is, when
        either centered vector collapses to the origin because every one of its
        values equals ``mu`` exactly (see Notes).

    Raises
    ------
    ValueError
        If the vectors differ in length, are empty, or contain ``NaN``.

    Notes
    -----
    GMCC is undefined only in the single degenerate case where all values of a
    vector coincide exactly with ``mu``.  A constant-valued profile whose value
    differs from ``mu`` stays well defined and evaluates to
    ``sign((c_x - mu) * (c_y - mu))`` (Proposition 3) -- precisely the case in
    which Pearson Correlation fails, since its centered vector is always zero
    for a constant profile.

    Examples
    --------
    Two constant profiles on opposite sides of the global mean:

    >>> round(gmcc([1, 1, 1], [5, 5, 5], mu=3.0), 3)
    -1.0

    A case where Pearson reports perfect positive correlation but the two
    entities actually sit on opposite sides of the population baseline:

    >>> round(gmcc([1, 2, 3], [3, 4, 5], mu=3.0), 3)
    -0.2

    Undefined: every value of ``x`` coincides with ``mu``.

    >>> gmcc([3, 3, 3], [1, 2, 3], mu=3.0)
    nan
    """
    xa = np.asarray(x, dtype=float).ravel()
    ya = np.asarray(y, dtype=float).ravel()

    if xa.shape != ya.shape:
        raise ValueError(
            f"x and y must have the same length; got {xa.size} and {ya.size}."
        )
    if xa.size == 0:
        raise ValueError("x and y must contain at least one co-observed value.")
    if np.isnan(xa).any() or np.isnan(ya).any():
        raise ValueError(
            "x and y must not contain NaN; pass only co-observed attributes "
            "(see gmcc_from_matrix for matrices with missing cells)."
        )

    xc = xa - mu
    yc = ya - mu

    norm_x = float(np.sqrt(np.dot(xc, xc)))
    norm_y = float(np.sqrt(np.dot(yc, yc)))

    # Undefined: a centered vector collapsed to the origin, i.e. every value of
    # that entity equals mu exactly.
    if norm_x < _EPS or norm_y < _EPS:
        return float("nan")

    value = float(np.dot(xc, yc) / (norm_x * norm_y))

    # Guard against floating-point drift just outside [-1, 1].
    return float(np.clip(value, -1.0, 1.0))


def gmcc_from_matrix(
    matrix: np.ndarray,
    i: int,
    j: int,
    mu: Optional[float] = None,
    min_co_observations: int = 1,
) -> float:
    """Compute GMCC between two rows of a matrix with missing values.

    Missing entries are marked by ``NaN``.  Only attributes observed by *both*
    rows enter the computation, matching the co-observation set ``I_uv`` of the
    paper.

    Parameters
    ----------
    matrix : ndarray of shape (n_entities, n_attributes)
        Observation matrix; ``NaN`` marks an unobserved entry.
    i, j : int
        Row indices of the two entities to compare.
    mu : float, optional
        The global mean.  If omitted it is computed from ``matrix`` itself.
        In a predictive setting, pass a ``mu`` computed on the training split
        so that no test information leaks into the similarity.
    min_co_observations : int, default 1
        Return ``nan`` when the two rows share fewer than this many observed
        attributes.  Raising it is a common safeguard under heavy sparsity.

    Returns
    -------
    float
        Similarity in ``[-1, 1]``, or ``nan`` if undefined or if the pair has
        too few co-observations.

    Examples
    --------
    >>> import numpy as np
    >>> R = np.array([[1.0, 1.0, 1.0, np.nan],
    ...               [5.0, 5.0, 5.0, 2.0]])
    >>> round(gmcc_from_matrix(R, 0, 1, mu=3.0), 3)
    -1.0
    """
    mat = np.asarray(matrix, dtype=float)
    if mat.ndim != 2:
        raise ValueError(f"matrix must be 2-dimensional; got {mat.ndim} dimensions.")
    if min_co_observations < 1:
        raise ValueError("min_co_observations must be at least 1.")

    if mu is None:
        mu = global_mean(mat)

    xa, ya = mat[i], mat[j]
    co_observed = ~np.isnan(xa) & ~np.isnan(ya)

    if int(co_observed.sum()) < min_co_observations:
        return float("nan")

    return gmcc(xa[co_observed], ya[co_observed], mu)


def gmcc_similarity_matrix(
    matrix: np.ndarray,
    mu: Optional[float] = None,
    min_co_observations: int = 1,
) -> np.ndarray:
    """Compute the full pairwise GMCC similarity matrix over the rows.

    The global mean is computed once and reused across every pair, which is the
    computational structure described in Section 3.4 of the paper.

    Parameters
    ----------
    matrix : ndarray of shape (n_entities, n_attributes)
        Observation matrix; ``NaN`` marks an unobserved entry.
    mu : float, optional
        The global mean.  Computed from ``matrix`` when omitted.
    min_co_observations : int, default 1
        Minimum number of shared observed attributes required for a pair.

    Returns
    -------
    ndarray of shape (n_entities, n_entities)
        Symmetric similarity matrix.  The diagonal holds the self-similarity,
        which is ``1.0`` wherever it is defined.  Undefined pairs hold ``nan``.

    Examples
    --------
    >>> import numpy as np
    >>> R = np.array([[1.0, 1.0, 1.0],
    ...               [5.0, 5.0, 5.0],
    ...               [1.0, 2.0, 3.0],
    ...               [3.0, 4.0, 5.0]])
    >>> S = gmcc_similarity_matrix(R)
    >>> round(float(S[0, 1]), 3)
    -1.0
    >>> round(float(S[2, 3]), 3)
    -0.2
    """
    mat = np.asarray(matrix, dtype=float)
    if mat.ndim != 2:
        raise ValueError(f"matrix must be 2-dimensional; got {mat.ndim} dimensions.")

    if mu is None:
        mu = global_mean(mat)

    n_entities = mat.shape[0]
    sim = np.full((n_entities, n_entities), np.nan, dtype=float)

    for a in range(n_entities):
        for b in range(a, n_entities):
            value = gmcc_from_matrix(
                mat, a, b, mu=mu, min_co_observations=min_co_observations
            )
            sim[a, b] = value
            sim[b, a] = value

    return sim


if __name__ == "__main__":  # pragma: no cover
    import doctest

    failures, _ = doctest.testmod()
    raise SystemExit(1 if failures else 0)
