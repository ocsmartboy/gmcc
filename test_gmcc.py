"""
Unit tests for the GMCC reference implementation.

Each test corresponds to a property established in the paper, so the suite
doubles as an executable check that the implementation matches the theory.

Run with::

    python -m pytest test_gmcc.py -v

or, without pytest installed::

    python test_gmcc.py
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from baselines import cosine, pearson
from gmcc import global_mean, gmcc, gmcc_from_matrix, gmcc_similarity_matrix

# A small rating matrix (scale 1-5) whose global mean is exactly 3.0.
TOY = np.array(
    [
        [1.0, 1.0, 1.0],  # U1: consistently low ratings
        [5.0, 5.0, 5.0],  # U2: consistently high ratings
        [1.0, 2.0, 3.0],  # U3: rising, below the population baseline
        [3.0, 4.0, 5.0],  # U4: rising, at or above the population baseline
    ]
)
MU = 3.0


# ----------------------------------------------------------------------
# Eq. (7): the global mean
# ----------------------------------------------------------------------


def test_global_mean_matches_equation_7():
    assert global_mean(TOY) == pytest.approx(MU)


def test_global_mean_ignores_missing_entries():
    with_missing = np.array([[1.0, np.nan], [5.0, 3.0]])
    assert global_mean(with_missing) == pytest.approx(3.0)


def test_global_mean_rejects_fully_missing_data():
    with pytest.raises(ValueError):
        global_mean([np.nan, np.nan])


# ----------------------------------------------------------------------
# Eq. (6): worked values
# ----------------------------------------------------------------------


def test_opposite_constant_profiles_are_maximally_dissimilar():
    # U1 and U2 sit symmetrically on opposite sides of mu.
    assert gmcc([1, 1, 1], [5, 5, 5], MU) == pytest.approx(-1.0)


def test_shifted_pair_disagrees_with_pearson_in_sign():
    # Pearson reports +1.0 here; GMCC sees the two entities straddling mu.
    assert gmcc([1, 2, 3], [3, 4, 5], MU) == pytest.approx(-0.2)
    assert pearson([1, 2, 3], [3, 4, 5]) == pytest.approx(1.0)


def test_zero_vector_is_defined_for_gmcc_but_not_for_cosine():
    # A genuinely observed all-zero profile: Cosine divides by zero, GMCC does not.
    assert math.isnan(cosine([3, 4, 5], [0, 0, 0]))
    assert gmcc([3, 4, 5], [0, 0, 0], mu=3.0) == pytest.approx(-0.7745966692414834)


# ----------------------------------------------------------------------
# Boundedness, symmetry and invariance (Section 4)
# ----------------------------------------------------------------------


def test_similarity_is_bounded():
    rng = np.random.default_rng(20260421)
    for _ in range(500):
        x = rng.uniform(1, 5, size=rng.integers(2, 12))
        y = rng.uniform(1, 5, size=x.size)
        value = gmcc(x, y, MU)
        if not math.isnan(value):
            assert -1.0 - 1e-9 <= value <= 1.0 + 1e-9


def test_similarity_is_symmetric():
    rng = np.random.default_rng(7)
    for _ in range(200):
        x = rng.uniform(1, 5, size=6)
        y = rng.uniform(1, 5, size=6)
        assert gmcc(x, y, MU) == pytest.approx(gmcc(y, x, MU))


def test_self_similarity_is_one_when_defined():
    assert gmcc([1, 2, 3], [1, 2, 3], MU) == pytest.approx(1.0)


def test_invariant_under_a_shared_shift_of_values_and_global_mean():
    # Shifting every value and mu by the same constant leaves the centered
    # vectors unchanged, so the similarity is unchanged.
    shift = 2.5
    base = gmcc([1, 2, 3], [3, 4, 5], MU)
    shifted = gmcc(
        [1 + shift, 2 + shift, 3 + shift],
        [3 + shift, 4 + shift, 5 + shift],
        MU + shift,
    )
    assert base == pytest.approx(shifted)


def test_invariant_under_positive_scaling():
    scale = 4.0
    base = gmcc([1, 2, 3], [3, 4, 5], MU)
    scaled = gmcc([1 * scale, 2 * scale, 3 * scale],
                  [3 * scale, 4 * scale, 5 * scale], MU * scale)
    assert base == pytest.approx(scaled)


# ----------------------------------------------------------------------
# Proposition 2: equivalence under aligned centering
# ----------------------------------------------------------------------


def test_equals_pearson_when_centering_references_are_aligned():
    rng = np.random.default_rng(123)
    for _ in range(200):
        x = rng.uniform(1, 5, size=8)
        y = rng.uniform(1, 5, size=8)
        # Aligned centering requires a single shared reference equal to each
        # entity's own local mean, so construct a pair that satisfies it.
        y = y - y.mean() + x.mean()
        aligned_mu = float(x.mean())
        assert gmcc(x, y, aligned_mu) == pytest.approx(pearson(x, y))


# ----------------------------------------------------------------------
# Proposition 3: definedness under constant profiles
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "c_x, c_y, expected_sign",
    [
        (1.0, 5.0, -1.0),  # opposite sides of mu
        (5.0, 1.0, -1.0),
        (1.0, 2.0, +1.0),  # both below mu
        (4.0, 5.0, +1.0),  # both above mu
    ],
)
def test_constant_profiles_follow_proposition_3(c_x, c_y, expected_sign):
    x = [c_x] * 4
    y = [c_y] * 4
    assert gmcc(x, y, MU) == pytest.approx(expected_sign)
    # Pearson is undefined for every one of these pairs.
    assert math.isnan(pearson(x, y))


def test_undefined_only_when_a_profile_coincides_with_the_global_mean():
    assert math.isnan(gmcc([MU, MU, MU], [1, 2, 3], MU))
    assert math.isnan(gmcc([1, 2, 3], [MU, MU, MU], MU))
    # One value off the global mean is enough to keep it defined.
    assert not math.isnan(gmcc([MU, MU, 4.0], [1, 2, 3], MU))


# ----------------------------------------------------------------------
# Matrix helpers and co-observation handling
# ----------------------------------------------------------------------


def test_matrix_helper_matches_the_vector_form():
    assert gmcc_from_matrix(TOY, 0, 1, mu=MU) == pytest.approx(
        gmcc(TOY[0], TOY[1], MU)
    )


def test_only_co_observed_attributes_are_used():
    matrix = np.array(
        [
            [1.0, 1.0, 1.0, np.nan],
            [5.0, 5.0, 5.0, 2.0],  # the 4th attribute is observed by one entity only
        ]
    )
    assert gmcc_from_matrix(matrix, 0, 1, mu=MU) == pytest.approx(-1.0)


def test_min_co_observations_filters_thin_pairs():
    matrix = np.array(
        [
            [1.0, np.nan, np.nan],
            [5.0, 5.0, 5.0],
        ]
    )
    assert not math.isnan(gmcc_from_matrix(matrix, 0, 1, mu=MU, min_co_observations=1))
    assert math.isnan(gmcc_from_matrix(matrix, 0, 1, mu=MU, min_co_observations=2))


def test_similarity_matrix_is_symmetric_with_unit_diagonal():
    sim = gmcc_similarity_matrix(TOY)
    assert sim.shape == (4, 4)
    np.testing.assert_allclose(sim, sim.T, equal_nan=True)
    for index in range(4):
        assert sim[index, index] == pytest.approx(1.0)


def test_similarity_matrix_reuses_a_single_global_mean():
    # Passing mu explicitly must give the same result as letting it be derived.
    np.testing.assert_allclose(
        gmcc_similarity_matrix(TOY),
        gmcc_similarity_matrix(TOY, mu=MU),
        equal_nan=True,
    )


# ----------------------------------------------------------------------
# Input validation
# ----------------------------------------------------------------------


def test_mismatched_lengths_are_rejected():
    with pytest.raises(ValueError):
        gmcc([1, 2, 3], [1, 2], MU)


def test_empty_vectors_are_rejected():
    with pytest.raises(ValueError):
        gmcc([], [], MU)


def test_nan_in_vectors_is_rejected():
    with pytest.raises(ValueError):
        gmcc([1.0, np.nan], [1.0, 2.0], MU)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-v"]))
