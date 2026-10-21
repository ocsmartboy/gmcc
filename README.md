# GMCC — Global Mean-Centered Correlation

Reference implementation of **GMCC**, a parameter-free, correlation-based similarity measure for sparse data environments, introduced in *Information Sciences* (Elsevier, 2026).

GMCC replaces the entity-specific **local** mean centering used by Pearson Correlation with a single population-level **global** mean, computed once over the dataset. This reduces the variance of the centering baseline by a factor of `n / N` and extends the definitional coverage of correlation-based similarity to cases where Pearson is undefined.

---

## The measure

For two value vectors `x` and `y` over `n` co-observed attributes, with global mean `μ` (Eq. 6):

```
                    Σᵢ (xᵢ − μ)(yᵢ − μ)
GMCC(x, y) = ──────────────────────────────────
             √(Σᵢ (xᵢ − μ)²) · √(Σᵢ (yᵢ − μ)²)
```

where the global mean is computed once over all `N` observed values `z_k` in the dataset (Eq. 7):

```
μ = (1/N) Σₖ zₖ
```

`μ` is **estimated from data, not tuned** — GMCC introduces no hyperparameter.

### What kind of data does this apply to?

**Any** dataset of entities characterised by partially observed, real-valued attributes. Following Section 3.1 of the paper, each observed value is "a measured quantity such as a rating, score, or feature value".

Nothing in the method assumes a rating scale, a bounded range, integer values, or a recommender-system setting. Because `μ` is estimated from whatever data you supply, the measure adapts to the scale of the domain on its own. Rows might be:

| Entities (rows) | Attributes (columns) | Values |
|---|---|---|
| Users | Items | Ratings |
| Documents | Terms | Counts or TF-IDF weights |
| Sensors or stations | Time points | Physical measurements |
| Samples | Genes or probes | Expression levels |
| Annotators | Items judged | Assigned scores |
| Respondents | Survey questions | Response values |

The examples below use ratings on a 1–5 scale **only because they are familiar and easy to read by hand** — [a worked example further down](#the-scale-is-not-assumed-a-non-rating-example) runs the same comparisons on an entirely different scale.

### How it relates to the classical measures

| | Centering reference | Computed from | Baseline variance |
|---|---|---|---|
| Cosine | none (origin at 0) | — | — |
| Pearson | local mean `x̄`, per entity | `n` co-observations | `σ²/n` |
| **GMCC** | **global mean `μ`, shared** | **`N` observations** | **`σ²/N`** |

Because `N ≫ n` under sparsity, the centering baseline is substantially more stable (Theorem 1).

---

## Installation

Requires Python 3.9+ and NumPy.

```bash
pip install numpy
```

The package is a small set of standalone modules — copy `gmcc.py` (and `baselines.py` if you want the comparison measures) into your project, or clone the repository.

---

## Quick start

```python
import numpy as np
from gmcc import global_mean, gmcc, gmcc_similarity_matrix

# Rows are entities, columns are attributes. Ratings here; any real-valued
# measurement works the same way.
ratings = np.array([
    [1.0, 1.0, 1.0],   # U1: consistently low ratings
    [5.0, 5.0, 5.0],   # U2: consistently high ratings
    [1.0, 2.0, 3.0],   # U3
    [3.0, 4.0, 5.0],   # U4
])

# Preprocessing — compute the global mean ONCE
mu = global_mean(ratings)          # 3.0

# Pairwise similarity, reusing mu
gmcc(ratings[0], ratings[1], mu)   # -1.0
gmcc(ratings[2], ratings[3], mu)   # -0.2

# Or the full similarity matrix
S = gmcc_similarity_matrix(ratings)
```

### Sparse data with missing entries

Mark unobserved entries with `NaN`. Only attributes observed by **both** entities enter the computation.

```python
ratings = np.array([
    [5.0, 3.0, np.nan, 1.0],
    [4.0, np.nan, 2.0, 1.0],
])

mu = global_mean(ratings)                      # NaN entries are ignored
S = gmcc_similarity_matrix(ratings, mu=mu, min_co_observations=2)
```

---

## Why it behaves differently

Open [`example.ipynb`](example.ipynb) to read this walkthrough with its output already rendered, or run `python example.py` to reproduce it in a terminal:

| Pair | Cosine | Pearson | GMCC | What is happening |
|---|---|---|---|---|
| `[1,1,1]` vs `[5,5,5]` | +1.000 | *undefined* | **−1.000** | Cosine sees identical direction and calls them identical; the two entities actually sit on opposite sides of `μ` |
| `[1,2,3]` vs `[3,4,5]` | +0.983 | +1.000 | **−0.200** | Pearson reports perfect correlation from the shared local pattern; GMCC sees one entity below and one above the population baseline |
| `[1,1,1]` vs `[1,2,3]` | +0.926 | *undefined* | **+0.775** | Pearson fails on the constant profile; GMCC reports that both sit below `μ` |
| `[3,4,5]` vs `[0,0,0]` | *undefined* | *undefined* | **−0.775** | An all-zero but genuinely observed profile breaks Cosine's normalization; GMCC remains defined |

### The scale is not assumed: a non-rating example

Four weather stations report temperature in °C at three time points. No ratings, no 1–5 scale, no recommender system.

```python
readings = np.array([
    [21.4, 22.1, 23.0],   # ST1: consistently cooler than the network
    [28.2, 29.0, 30.1],   # ST2: consistently warmer than the network
    [21.0, 25.3, 29.4],   # ST3: swings across the network mean
    [25.2, 25.2, 25.2],   # ST4: flat, close to the network mean
])

mu = global_mean(readings)   # 25.425 -- derived from the data, not assumed
```

| Pair | Cosine | Pearson | GMCC | |
|---|---|---|---|---|
| ST1 vs ST2 | +1.000 | +1.000 | **−0.918** | Both baselines call these a perfect match; both are misled by the shared upward trend. GMCC registers that ST1 sits below the network mean throughout while ST2 sits above it. |
| ST1 vs ST4 | +1.000 | *undefined* | **+0.980** | ST4 is flat, so Pearson has no variance to work with. |
| ST3 vs ST4 | +0.991 | *undefined* | **+0.056** | Cosine sees near-identity; GMCC reports almost no relationship, since ST3 straddles the mean that ST4 sits on. |

This is the same behaviour as in the rating examples, on a completely different scale — because `μ` is fitted to the data rather than fixed by the domain.

### When is GMCC undefined?

Only in one degenerate case: when **every** value of an entity equals `μ` exactly, so its centered vector collapses to the origin. A constant-valued profile whose value differs from `μ` stays well defined and evaluates to `sign((c_x − μ)(c_y − μ))` — this is Proposition 3, and it is exactly the case where Pearson always fails.

---

## Repository contents

| File | Purpose |
|---|---|
| `gmcc.py` | The measure itself — the only file you need to use it |
| `baselines.py` | Cosine and Pearson, for side-by-side comparison |
| `example.ipynb` | Annotated walkthrough with outputs saved (readable on GitHub without running) |
| `example.py` | The same examples as a runnable script |
| `test_gmcc.py` | Test suite checking the implementation against the paper's propositions |

`example.ipynb` additionally requires `pandas` for table formatting; `gmcc.py` itself needs only NumPy.

---

## API

| Function | Purpose |
|---|---|
| `global_mean(data)` | Global mean `μ` over all observed values (Eq. 7). `NaN` entries are ignored. |
| `gmcc(x, y, mu)` | GMCC between two aligned, co-observed vectors (Eq. 6). |
| `gmcc_from_matrix(matrix, i, j, mu=None, min_co_observations=1)` | GMCC between two rows of a matrix with `NaN` for missing entries. |
| `gmcc_similarity_matrix(matrix, mu=None, min_co_observations=1)` | Full pairwise similarity matrix over the rows. |

`baselines.py` additionally provides `cosine(x, y)` and `pearson(x, y)` for side-by-side comparison. All functions return `nan` where the measure is undefined.

### Two usage notes

**Compute `μ` once.** It is a property of the dataset, not of a pair. Computing it per pair defeats the variance reduction that motivates the measure.

**In a predictive setting, fit `μ` on the training split only.** Computing it over the full dataset lets test information leak into the similarity computation:

```python
mu = global_mean(train_matrix)                     # fit on training data
S = gmcc_similarity_matrix(full_matrix, mu=mu)     # then apply
```

---

## Complexity

Identical in order to Cosine and Pearson (Section 3.4):

- **Preprocessing:** `O(N)`, a single pass, done once.
- **Per pair:** `O(n)`, the same as Pearson and Cosine.
- **Extra storage:** `O(1)` — one scalar. Pearson requires `O(|U|)` to store or recompute per-entity statistics.

---

## Tests

```bash
pip install pytest
python -m pytest test_gmcc.py -v
```

The suite checks the properties established in the paper against the implementation: boundedness, symmetry, invariance to shared shift and positive scaling, equivalence with Pearson under aligned centering (Proposition 2), definedness under constant profiles (Proposition 3), and co-observation handling.

---

## Citation

If you use this code, please cite the paper:

```bibtex
@article{wiyono2026gmcc,
  title   = {Global mean--centered correlation: A variance-reduced and
             theoretically grounded similarity measure for sparse data environments},
  author  = {Wiyono, Slamet and Adji, Teguh Bharata and
             Nugroho, Hanung Adi and Wibirama, Sunu},
  journal = {Information Sciences},
  volume  = {749},
  pages   = {123502},
  year    = {2026},
  doi     = {10.1016/j.ins.2026.123502},
  issn    = {0020-0255},
  url     = {https://doi.org/10.1016/j.ins.2026.123502}
}
```

---

## License

MIT — see [LICENSE](LICENSE).

The license covers this implementation. The paper itself is published by Elsevier under its own terms.
