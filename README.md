# Filtered Approximate Nearest Neighbor Search (ANNS)

A from-scratch implementation of two filtered vector search methods described in:

> *Filtered Vector Search: State-of-the-art and Research Opportunities*  
> Chronis et al., PVLDB 18(12), 2025

---

## What is Filtered Vector Search?

Standard vector search finds the *k* nearest vectors to a query. **Filtered** vector search adds a relational constraint: only vectors whose attribute value falls within a range `[lo, hi]` are eligible. For example:

> *"Find the 10 most similar products to this image, but only among items priced between \$20 and \$50."*

The challenge is that naively applying a filter either degrades recall (post-filtering discards too many ANN candidates) or kills speed (pre-filtering shrinks the index too aggressively). This codebase implements and evaluates both approaches.

---

## Methods Implemented

### A — Pre-filtering (Exact KNN baseline)
```
Filter base vectors by label → exact KNN on the surviving subset
```
- **Exact**: recall is always 1.0 by definition, so it doubles as **ground truth**
- **Best when**: the filter is highly selective (small surviving subset)
- **Worst when**: the filter passes most vectors (expensive KNN on large set)

### B — Post-filtering (LSH + label filter)
```
LSH ANN on all vectors → discard candidates outside [lo, hi] → rerank survivors
```
- **Approximate**: recall depends on LSH parameters
- **Best when**: the filter is loose (many vectors survive, few candidates wasted)
- **Worst when**: the filter is tight (most LSH candidates discarded, recall drops)
- ANN index used: **E2LSH** (random projection LSH for Euclidean distance)

---

## File Structure

```
.
├── main.py          # Entry point — runs both methods, prints comparison table
├── prefilter.py     # Method A: filter-then-KNN (exact, ground truth)
├── postfilter.py    # Method B: LSH ANN then post-filter
├── lsh_index.py     # E2LSH implementation (optimised)
├── data_utils.py    # SIFT loader, label assignment, filter-range generation
├── evaluate.py      # Recall@K, QPS, per-selectivity breakdown
├── lsh_explain.md   # Detailed explanation of the LSH implementation
└── README.md        # This file
```

---

## Dataset

This project uses the **SIFT-128** dataset from the ANN-benchmarks suite.

| Split | Shape | Description |
|---|---|---|
| `train` | (1 000 000, 128) | Database vectors |
| `test`  | (10 000, 128)   | Query vectors |

**Download:**
```bash
mkdir -p data
wget -P data https://storage.googleapis.com/ann-datasets/ann-benchmarks/sift-128-euclidean.hdf5
```

Place the file at `./data/sift-128-euclidean.hdf5`. The loader normalises all vectors to `[0, 1]` by dividing by 255.

**Labels and filter ranges** are not part of the original dataset — they are generated synthetically:
- Each base vector is assigned a random integer label in `[0, n_labels)`
- Each query is assigned a random range `[lo, hi]` controlling filter selectivity

---

## Quickstart

```bash
pip install -r requirements.txt

# Run with SIFT data (default settings)
python main.py --sift

# Vary key parameters
python main.py --lsh-tables 200 --lsh-bin-width 0.8
```

---

## CLI Reference

| Argument | Default | Description |
|---|---|---|
| `--sift` | `False` | Use SIFT dataset (required for real experiments) |
| `--sift-dir` | `./data` | Directory containing the `.hdf5` file |
| `--max-base` | `None` (all) | Limit base vectors loaded |
| `--n-query` | `None` (all) | Limit query vectors used |
| `--n-labels` | `1000` | Number of distinct integer label values |
| `--min-sel` | `0.05` | Minimum filter selectivity (fraction of label space) |
| `--max-sel` | `0.40` | Maximum filter selectivity |
| `--k` | `50` | Number of nearest neighbors to retrieve |
| `--lsh-tables` | `400` | LSH: number of hash tables *L* |
| `--lsh-functions` | `5` | LSH: hash functions per table *K* |
| `--lsh-bin-width` | `0.9` | LSH: projection bin width *w* (key recall knob) |
| `--seed` | `42` | Global random seed |

---

## Evaluation Metrics

**Recall@K** — fraction of the true top-K neighbors found:

$$\text{Recall@K} = \frac{|R \cap G|}{|G|}$$

where *G* is the exact ground truth (from pre-filtering) and *R* is the ANN result. Averaged over all queries.

**QPS (Queries Per Second)** — search throughput, wall-clock time, single thread. Index build time is excluded.

The output table also breaks down recall by **filter selectivity bin**, showing how post-filter recall degrades as filters become tighter — the core phenomenon discussed in the paper.

---

## Tuning the LSH Index

Three parameters control the recall–speed tradeoff:

| Parameter | Effect on recall | Effect on speed |
|---|---|---|
| `--lsh-tables` *L* | ↑ more tables → higher recall | ↓ more dict lookups |
| `--lsh-functions` *K* | ↓ more functions → smaller buckets, lower per-table recall | ↑ fewer candidates to rerank |
| `--lsh-bin-width` *w* | ↑ larger bins → more collisions, higher recall | ↓ more candidates to filter |

The miss probability for a true neighbor at distance *d* is:

$$P(\text{miss}) = (1 - p(d)^K)^L$$

where $p(d)$ is the single-function collision probability. See `lsh_explain.md` for the full derivation and the three implementation optimisations that achieve **17× speedup** over a naïve baseline.

---

## Performance (100K × 128 SIFT vectors, 1K queries)

| Method | Recall@10 | QPS |
|---|---|---|
| Pre-filter (exact) | 1.000 | ~200–400 |
| Post-filter (LSH, tuned) | ~0.85–0.95 | ~3 000–5 000 |

Post-filtering is 10–15× faster than pre-filtering at moderate recall. Recall drops for highly selective filters — see the selectivity breakdown in the output.

---

## Extending the Code

The codebase is structured so each component is independently swappable:

- **Different ANN index**: replace `lsh_index.py` with HNSW, IVF, etc. and plug it into `PostFilterSearch.__init__()` — the `batch_query()` interface is the only contract.
- **Different filter types**: `prefilter.py` and `postfilter.py` only use `labels`, `lo`, and `hi` — swap in multi-attribute or categorical filters by changing `_filter_and_rerank()`.
- **Different datasets**: replace `load_sift()` in `data_utils.py`.
- **Additional metrics**: add functions to `evaluate.py` following the `compute_recall` pattern.

---

## References

- Chronis et al. (2025). *Filtered Vector Search: State-of-the-art and Research Opportunities.* PVLDB 18(12).
- Datar et al. (2004). *Locality-Sensitive Hashing Scheme Based on p-Stable Distributions.* SoCG.
- Malkov & Yashunin (2020). *Efficient and Robust ANN Search Using HNSW.* IEEE TPAMI.
- ANN Benchmarks: https://ann-benchmarks.com
