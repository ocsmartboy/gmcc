"""
Baseline similarity measures, for comparison against GMCC.

These are the two classical correlation-based measures discussed in Section 3.1
of the paper.  They are provided so the behavioural differences reported there
can be reproduced side by side; GMCC itself lives in ``gmcc.py``.

    Cosine (Eq. 4):   sum x_i y_i / (||x|| ||y||)
    Pearson (Eq. 5):  sum (x_i - x_bar)(y_i - y_bar)
                      / (sqrt(sum (x_i - x_bar)^2) sqrt(sum (y_i - y_bar)^2))

Both return ``nan`` where they are undefined:

* Cosine, when either vector is the zero vector.
* Pearson, whenever either vector is constant (zero variance) -- the case
  GMCC is designed to cover (Proposition 3).

License: MIT
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

__all__ = ["cosine", "pearson"]

_EPS = 1e-12


def _as_pair(
    x: Sequence[float] | np.ndarray, y: Sequence[float] | np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    xa = np.asarray(x, dtype=float).ravel()
    ya = np.asarray(y, dtype=float).ravel()
    if xa.shape != ya.shape:
        raise ValueError(
            f"x and y must have the same length; got {xa.size} and {ya.size}."
        )
    if xa.size == 0:
        raise ValueError("x and y must contain at least one value.")
    if np.isnan(xa).any() or np.isnan(ya).any():
        raise ValueError("x and y must not contain NaN.")
    return xa, ya


def cosine(
    x: Sequence[float] | np.ndarray, y: Sequence[float] | np.ndarray
) -> float:
    """Cosine Similarity (Eq. 4).

    Returns ``nan`` when either vector is the zero vector.

    Examples
    --------
    >>> round(cosine([1, 1, 1], [5, 5, 5]), 3)
    1.0
    >>> cosine([3, 4, 5], [0, 0, 0])
    nan
    """
    xa, ya = _as_pair(x, y)
    norm_x = float(np.sqrt(np.dot(xa, xa)))
    norm_y = float(np.sqrt(np.dot(ya, ya)))
    if norm_x < _EPS or norm_y < _EPS:
        return float("nan")
    return float(np.clip(np.dot(xa, ya) / (norm_x * norm_y), -1.0, 1.0))


def pearson(
    x: Sequence[float] | np.ndarray, y: Sequence[float] | np.ndarray
) -> float:
    """Pearson Correlation with local mean centering (Eq. 5).

    Returns ``nan`` when either vector has zero variance, i.e. for any
    constant-valued profile.

    Examples
    --------
    >>> round(pearson([1, 2, 3], [3, 4, 5]), 3)
    1.0
    >>> pearson([1, 1, 1], [5, 5, 5])
    nan
    """
    xa, ya = _as_pair(x, y)
    xc = xa - xa.mean()
    yc = ya - ya.mean()
    norm_x = float(np.sqrt(np.dot(xc, xc)))
    norm_y = float(np.sqrt(np.dot(yc, yc)))
    if norm_x < _EPS or norm_y < _EPS:
        return float("nan")
    return float(np.clip(np.dot(xc, yc) / (norm_x * norm_y), -1.0, 1.0))


if __name__ == "__main__":  # pragma: no cover
    import doctest

    failures, _ = doctest.testmod()
    raise SystemExit(1 if failures else 0)
