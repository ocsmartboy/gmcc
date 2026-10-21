"""
Worked examples for GMCC.

Run with::

    python example.py

Every number printed below is computed at run time -- nothing is hard-coded --
so the output doubles as a check that the installation behaves as documented.

Examples 1-3 use ratings on a 1-5 scale because they are familiar and easy to
follow by hand.  GMCC assumes no rating scale: example 4 runs the same
comparisons on sensor measurements in degrees Celsius.
"""

from __future__ import annotations

import numpy as np

from baselines import cosine, pearson
from gmcc import global_mean, gmcc, gmcc_similarity_matrix


def fmt(value: float) -> str:
    """Format a similarity value, showing undefined results explicitly."""
    return "undefined" if np.isnan(value) else f"{value:+.3f}"


def example_1_global_mean() -> float:
    """The one-time preprocessing step: compute the global mean."""
    print("=" * 72)
    print("1. GLOBAL MEAN (Eq. 7) -- computed once, reused for every pair")
    print("=" * 72)

    ratings = np.array(
        [
            [1.0, 1.0, 1.0],  # U1: consistently low ratings
            [5.0, 5.0, 5.0],  # U2: consistently high ratings
            [1.0, 2.0, 3.0],  # U3: rising, below the population baseline
            [3.0, 4.0, 5.0],  # U4: rising, at or above the population baseline
        ]
    )

    print("\n    Rating matrix (4 users x 3 items, scale 1-5):\n")
    print("          " + "".join(f"{name:>8}" for name in ("Item A", "Item B", "Item C")))
    for index, row in enumerate(ratings, start=1):
        print(f"    U{index}    " + "".join(f"{value:8.0f}" for value in row))

    mu = global_mean(ratings)
    print(f"\n    mu = sum of all 12 observed values / 12 = {mu:.4f}")
    return mu


def example_2_pairwise(mu: float) -> None:
    """Side-by-side behaviour of the three measures on telling cases."""
    print()
    print("=" * 72)
    print(f"2. PAIRWISE COMPARISON (mu = {mu:.1f})")
    print("=" * 72)

    cases = [
        ("U1 vs U2", [1, 1, 1], [5, 5, 5],
         "constant profiles on opposite sides of mu"),
        ("U3 vs U4", [1, 2, 3], [3, 4, 5],
         "same rising pattern, different level"),
        ("U1 vs U3", [1, 1, 1], [1, 2, 3],
         "both below the population baseline"),
        ("U2 vs U2", [5, 5, 5], [5, 5, 5],
         "identical constant profiles"),
        ("zero row ", [3, 4, 5], [0, 0, 0],
         "an all-zero but genuinely observed profile"),
    ]

    print(f"\n    {'Pair':<10} {'Cosine':>10} {'Pearson':>10} {'GMCC':>10}   Notes")
    print("    " + "-" * 68)
    for label, x, y, note in cases:
        print(
            f"    {label:<10} {fmt(cosine(x, y)):>10} "
            f"{fmt(pearson(x, y)):>10} {fmt(gmcc(x, y, mu)):>10}   {note}"
        )

    print("\n    Reading the table:")
    print("    - Cosine ignores baseline differences, so U1 vs U2 looks identical.")
    print("    - Pearson is undefined for every constant profile (zero variance).")
    print("    - Cosine is also undefined against an all-zero vector.")
    print("    - GMCC stays defined throughout, because it centers on mu rather")
    print("      than on each entity's own mean (Proposition 3).")


def example_3_sparse_matrix() -> None:
    """Sparse data with missing entries, and the full similarity matrix."""
    print()
    print("=" * 72)
    print("3. SPARSE MATRIX -- missing entries marked NaN")
    print("=" * 72)

    ratings = np.array(
        [
            [5.0, 3.0, np.nan, 1.0],
            [4.0, np.nan, 2.0, 1.0],
            [np.nan, 5.0, 4.0, np.nan],
            [1.0, 1.0, 2.0, 5.0],
        ]
    )

    print("\n    Rating matrix (4 users x 4 items):\n")
    print("          " + "".join(f"{name:>8}" for name in ("I1", "I2", "I3", "I4")))
    for index, row in enumerate(ratings, start=1):
        cells = "".join(
            f"{'--':>8}" if np.isnan(value) else f"{value:8.0f}" for value in row
        )
        print(f"    U{index}    {cells}")

    mu = global_mean(ratings)
    n_observed = int(np.sum(~np.isnan(ratings)))
    print(f"\n    mu over the {n_observed} observed values = {mu:.4f}")

    similarity = gmcc_similarity_matrix(ratings, mu=mu, min_co_observations=2)
    print("\n    GMCC similarity matrix (pairs with < 2 co-observations -> nan):\n")
    print("          " + "".join(f"{name:>8}" for name in ("U1", "U2", "U3", "U4")))
    for index, row in enumerate(similarity, start=1):
        cells = "".join(
            f"{'--':>8}" if np.isnan(value) else f"{value:+8.3f}" for value in row
        )
        print(f"    U{index}    {cells}")


def example_4_non_rating_domain() -> None:
    """The measure assumes no rating scale: the same behaviour on sensor data."""
    print()
    print("=" * 72)
    print("4. A NON-RATING DOMAIN -- no 1-5 scale, no recommender system")
    print("=" * 72)

    # Four weather stations reporting temperature in degrees Celsius.
    readings = np.array(
        [
            [21.4, 22.1, 23.0],  # ST1: consistently cooler than the network
            [28.2, 29.0, 30.1],  # ST2: consistently warmer than the network
            [21.0, 25.3, 29.4],  # ST3: swings across the network mean
            [25.2, 25.2, 25.2],  # ST4: flat, close to the network mean
        ]
    )
    stations = ["ST1", "ST2", "ST3", "ST4"]

    print("\n    Temperature readings (degrees C) at three time points:\n")
    print("          " + "".join(f"{name:>8}" for name in ("t1", "t2", "t3")))
    for name, row in zip(stations, readings):
        print(f"    {name}   " + "".join(f"{value:8.1f}" for value in row))

    mu = global_mean(readings)
    print(f"\n    mu = {mu:.3f} degrees C -- derived from the data, not assumed")

    pairs = [
        (0, 1, "both baselines are misled by the shared upward trend"),
        (0, 3, "ST4 is flat, so Pearson has no variance to work with"),
        (2, 3, "ST3 straddles the mean that ST4 sits on"),
    ]

    print(f"\n    {'Pair':<14} {'Cosine':>10} {'Pearson':>10} {'GMCC':>10}   Notes")
    print("    " + "-" * 68)
    for a, b, note in pairs:
        label = f"{stations[a]} vs {stations[b]}"
        print(
            f"    {label:<14} {fmt(cosine(readings[a], readings[b])):>10} "
            f"{fmt(pearson(readings[a], readings[b])):>10} "
            f"{fmt(gmcc(readings[a], readings[b], mu)):>10}   {note}"
        )

    print("\n    Same behaviour as the rating examples, on a different scale --")
    print("    because mu is fitted to the data rather than fixed by the domain.")


def example_5_train_test_discipline() -> None:
    """In a predictive setting, mu must come from the training split only."""
    print()
    print("=" * 72)
    print("5. TRAIN/TEST DISCIPLINE")
    print("=" * 72)

    train = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 3.0]])
    test = np.array([[5.0, 5.0, 5.0], [4.0, 5.0, 4.0]])

    mu_train = global_mean(train)
    mu_all = global_mean(np.vstack([train, test]))

    print(f"\n    mu from the training split only : {mu_train:.4f}   <- use this")
    print(f"    mu from train + test together   : {mu_all:.4f}   <- leaks test data")
    print("\n    Computing mu over the full dataset lets test information reach")
    print("    the similarity computation. Fit it on the training split, then")
    print("    pass that value explicitly:")
    print("\n        mu = global_mean(train_matrix)")
    print("        sim = gmcc_similarity_matrix(full_matrix, mu=mu)")


def main() -> None:
    mu = example_1_global_mean()
    example_2_pairwise(mu)
    example_3_sparse_matrix()
    example_4_non_rating_domain()
    example_5_train_test_discipline()
    print()
    print("=" * 72)
    print("Reference: Wiyono, Adji, Nugroho & Wibirama (2026),")
    print("Information Sciences 749, 123502.")
    print("https://doi.org/10.1016/j.ins.2026.123502")
    print("=" * 72)


if __name__ == "__main__":
    main()
