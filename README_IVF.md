# Maximizing the Filtered-ANN Score — Step-by-Step Guide

This guide walks you through **why** the baseline scores low and **how** to beat
it, end to end. Follow the steps in order; each one is copy-paste runnable on
your local machine.

The competition metric is:

```
S = (QPS / 100) * Recall@50 ** 2
```

- **QPS** = queries per second (search only; index build time excluded)
- **Recall@50** = fraction of the true filtered top-50 neighbors found
- Recall enters **squared**, but it's capped at 1.0 — so once recall is high,
  the only way to keep climbing is **more QPS**.

---

## Table of Contents

1. [The key insight (read this first)](#1-the-key-insight)
2. [What I built for you](#2-what-i-built-for-you)
3. [Step 1 — Environment setup](#3-step-1--environment-setup)
4. [Step 2 — Download the dataset](#4-step-2--download-the-dataset)
5. [Step 3 — Reproduce the LSH baseline](#5-step-3--reproduce-the-lsh-baseline)
6. [Step 4 — Run the IVF sweep to find best parameters](#6-step-4--run-the-ivf-sweep)
7. [Step 5 — Wire the winner into main.py](#7-step-5--wire-the-winner-into-mainpy)
8. [Step 6 — (Optional) Squeeze out more QPS](#8-step-6--optional-squeeze-out-more-qps)
9. [How the IVF method works](#9-how-the-ivf-method-works)
10. [Tuning cheatsheet](#10-tuning-cheatsheet)
11. [Troubleshooting](#11-troubleshooting)

---

## 1. The Key Insight

Your full-scale run (1M base, 10 000 queries, K=50) produced:

| Method            | Recall | QPS   | **Score** |
|-------------------|--------|-------|-----------|
| PostFilter (LSH)  | 0.704  | 116.3 | **0.58**  |
| PreFilter (exact) | 1.000  | 16.0  | 0.16      |

**The score is QPS-bound, not recall-bound.**

- Pushing recall 0.70 → 1.00 multiplies the score by at most `(1/0.70)² ≈ 2.0×`.
- But QPS is only **116**. The LSH index buys just **7×** over brute force —
  that's where 10×+ is hiding.

Why is LSH so slow? Its query path is a **Python loop** over 10 000 queries, and
inside each it does **L = 400 dictionary lookups + a sort + a dedup**. That's
~4 million dict lookups per benchmark. The matmul is ~5% of the time; the Python
loop is ~95%.

**Strategy: replace LSH with an IVF (inverted-file) index** whose query path is
one matmul + a few small, label-sorted slice gathers. Expect QPS in the
**several hundred to >1000** range at recall ~0.85+, pushing the score from
**0.58 toward 3–8+**.

---

## 2. What I Built For You

Three new files (already in this folder):

| File             | Purpose |
|------------------|---------|
| `ivf_index.py`   | The IVF index: NumPy k-means + label-sorted inverted lists + fast filtered search. |
| `ivf_search.py`  | A thin wrapper with the **same interface** as `PostFilterSearch`, so it drops into `main.py` with a one-line import change. |
| `sweep_ivf.py`   | A stand-alone benchmark that builds the exact ground truth once, then sweeps `n_centroids × nprobe` and prints the score for each — highlighting the best. |

Nothing else is modified, so your original LSH code still works for comparison.

---

## 3. Step 1 — Environment Setup

```bash
# from the project root
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt    # numpy, h5py, matplotlib
```

Verify:

```bash
python3 -c "import numpy, h5py, matplotlib; print('deps OK')"
```

> **RAM note:** the exact ground-truth pass on the full 1M set needs roughly
> **4–6 GB** of free RAM. If you have less, use the `--max-base` flag shown in
> Step 4 to work on a subset first.

---

## 4. Step 2 — Download the Dataset

```bash
mkdir -p data
wget -P data https://storage.googleapis.com/ann-datasets/ann-benchmarks/sift-128-euclidean.hdf5
```

This is the standard **SIFT-128** ANN-benchmarks file:
- `train`: 1 000 000 × 128 (base vectors)
- `test` : 10 000 × 128 (queries)

The loaders divide by 255 to normalize values into `[0, 1]`.

---

## 5. Step 3 — Reproduce the LSH Baseline

So you have a number to beat:

```bash
python3 main.py --sift
```

You should see something close to the table in Section 1 (Score ≈ 0.58). This
confirms your environment matches the competition setup.

> If you hit an out-of-memory error here, reduce the working set:
> ```bash
> python3 main.py --sift --max-base 200000 --n-query 2000
> ```
> (The score scale will differ from the full run, but it lets you iterate fast.)

---

## 6. Step 4 — Run the IVF Sweep

This is the main event. It finds the parameters that maximize the score.

**Full scale (matches the competition):**

```bash
python3 sweep_ivf.py
```

**Faster smoke test on a subset (recommended first, ~1 minute):**

```bash
python3 sweep_ivf.py --max-base 200000 --n-query 2000
```

**Custom grid:**

```bash
python3 sweep_ivf.py --centroids 2048 4096 8192 --nprobe 8 16 24 32 48 64
```

Example output (numbers illustrative — yours will vary by machine):

```
================================================================
  centroids  nprobe  Recall@K        QPS     SCORE
================================================================
       4096       8    0.7421     1850.3    1.0192  <- best so far
       4096      16    0.8633     1120.7    0.8352
       4096      24    0.9105      790.4    0.6553
       4096      32    0.9398      610.2    0.5390
       4096      48    0.9655      415.8    0.3877
================================================================

BEST: n_centroids=4096  nprobe=8  Recall@50=0.7421  QPS=1850.3  SCORE=1.0192
```

**Read the table like this:**
- As `nprobe` ↑, recall ↑ but QPS ↓.
- Because recall is squared *and capped at 1*, the score usually peaks at a
  **moderate** recall (≈0.80–0.95), **not** at the highest recall.
- The script prints the single best `(n_centroids, nprobe)` at the end. Copy it.

> **Tip:** if the best row is at the **edge** of your grid (e.g. lowest or
> highest `nprobe`), widen the grid in that direction and re-run.

---

## 7. Step 5 — Wire the Winner into main.py

Open `main.py` and make **two edits**.

**(a) Change the import.** Find:

```python
from postfilter   import PostFilterSearch
```

Replace with:

```python
# from postfilter import PostFilterSearch
from ivf_search   import IVFFilteredSearch as PostFilterSearch
```

**(b) Change the construction.** Find the block:

```python
post = PostFilterSearch(
    base_vecs,
    labels,
    k_multiplier = args.k_multiplier,
    n_tables     = args.lsh_tables,
    n_functions  = args.lsh_functions,
    bin_width    = args.lsh_bin_width,
    seed         = args.seed,
    **filter_aug_params,
)
```

Replace with (use the numbers the sweep reported as BEST):

```python
post = PostFilterSearch(
    base_vecs,
    labels,
    n_centroids = 4096,   # <- from sweep_ivf.py BEST
    nprobe      = 8,      # <- from sweep_ivf.py BEST
    seed        = args.seed,
)
```

> The wrapper ignores any leftover LSH kwargs, so even if you forget to remove
> `n_tables`/`bin_width`/`**filter_aug_params`, it won't crash — but cleaner is
> better.

Now run:

```bash
python3 main.py --sift
```

The comparison table will print your new IVF score, and `final_score.png` will
be regenerated with the new point.

---

## 8. Step 6 — (Optional) Squeeze Out More QPS

If you want to push the score even higher, here are the next levers in order of
effort vs. reward:

1. **Cast the base matrix to `float32` and ensure it's contiguous** (already
   done in `ivf_index.py`). Make sure NumPy is using a fast BLAS (OpenBLAS/MKL):
   ```bash
   python3 -c "import numpy as np; np.show_config()"
   ```
   If you see a reference/`generic` BLAS, install one:
   `pip install numpy` from a wheel that bundles OpenBLAS, or `conda install mkl`.

2. **Raise `n_centroids`** (e.g. 8192 or 16384) so each list is smaller. Smaller
   lists = fewer vectors to rerank per query = higher QPS. You'll need a slightly
   higher `nprobe` to hold recall, so re-sweep.

3. **Multi-threaded queries.** The benchmark loop is single-threaded by design,
   but if the competition allows threads, wrap the per-query work in a
   `concurrent.futures.ThreadPoolExecutor` (NumPy releases the GIL during the
   rerank matmul). This can give a near-linear speedup on multi-core machines.

4. **Fully vectorized batch rerank.** Replace the Python per-query loop entirely
   by grouping queries that probe the same lists and reranking in bulk. Bigger
   rewrite, removes the last Python bottleneck; ask if you want this version.

---

## 9. How the IVF Method Works

**Build (once):**
1. Train `n_centroids` cluster centers with NumPy k-means (on a subsample for
   speed).
2. Assign every base vector to its nearest centroid → an **inverted list** per
   centroid (disjoint Voronoi cells).
3. Store each list's members **sorted by label**, in flat CSR-style arrays for
   cache-friendly gathers.

**Search (per batch):**
1. **One matmul** `(Q, D) @ (D, C)` scores all queries against all centroids.
2. Pick the `nprobe` nearest lists per query (`argpartition`).
3. For each query, for each probed list: the list is **label-sorted**, so the
   filter `label ∈ [lo, hi]` becomes **two binary searches** (`searchsorted`) —
   we slice out exactly the in-range rows and never touch the rest.
4. Concatenate those slices (no dedup needed — Voronoi cells are disjoint),
   rerank by exact L2, return top-k.

**Why it beats LSH for this score:**

| | LSH (baseline) | IVF (this method) |
|---|---|---|
| Query path | 400 dict lookups + sort + dedup / query | 1 matmul + a few slice gathers |
| Candidate set | huge; ~70% discarded by filter | bounded; **label-sorted slices = zero waste** |
| Recall knob | L, K, w (coupled, fiddly) | just `nprobe` (monotonic, easy to sweep) |

The two ideas doing the heavy lifting are **disjoint inverted lists** (no
cross-table dedup) and **label-sorted lists + `searchsorted`** (the filter costs
two binary searches instead of scanning + masking).

---

## 10. Tuning Cheatsheet

| Parameter     | Effect on Recall | Effect on QPS | Guidance |
|---------------|------------------|---------------|----------|
| `nprobe`      | ↑ higher         | ↓ lower       | **The main knob.** Sweep it; score usually peaks at recall ≈ 0.80–0.95. |
| `n_centroids` | indirect         | ↑ (smaller lists) | `sqrt(N)`–`4·sqrt(N)`. For N=1M try 2048–8192. More centroids → raise `nprobe`. |
| `n_iter`      | tiny             | build only    | 5–10 is plenty; affects only build time, not search. |

Rule of thumb for N = 1 000 000: start at `n_centroids=4096`, sweep
`nprobe ∈ {8,16,24,32,48}`, then refine around the winner.

---

## 11. Troubleshooting

**`MemoryError` / process killed during ground truth.**
The exact pre-filter on 1M vectors is memory-heavy. Use a subset:
`python3 sweep_ivf.py --max-base 200000 --n-query 2000`. Once you know good
parameters on the subset, validate them once on the full set.

**Recall is lower than expected.**
Increase `nprobe`, or increase `n_centroids` and re-sweep. Also confirm you're
using the same `n_labels`, `min_sel`, `max_sel`, and `seed` as the official
benchmark (defaults here match `main.py`).

**QPS is lower than expected.**
Check your BLAS (Section 8.1). A reference BLAS can be 5–10× slower on the matmul
and rerank. Also make sure you're timing **search only** (the harness already
excludes build time).

**`FileNotFoundError: sift-128-euclidean.hdf5`.**
Re-run Step 2, or pass `--sift-dir /path/to/folder` to `sweep_ivf.py`.

---

## Quick Recap (TL;DR)

```bash
# 1. setup
pip install -r requirements.txt

# 2. data
mkdir -p data && wget -P data \
  https://storage.googleapis.com/ann-datasets/ann-benchmarks/sift-128-euclidean.hdf5

# 3. baseline to beat
python3 main.py --sift

# 4. find best IVF params
python3 sweep_ivf.py                       # or --max-base 200000 --n-query 2000

# 5. edit main.py:
#    from ivf_search import IVFFilteredSearch as PostFilterSearch
#    post = PostFilterSearch(base_vecs, labels, n_centroids=<BEST>, nprobe=<BEST>)
python3 main.py --sift                      # confirm the new, higher score
```
