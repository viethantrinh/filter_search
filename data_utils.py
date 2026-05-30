"""
data_utils.py
=============
Utilities for loading the SIFT dataset from disk (standard .fvecs / .ivecs
format) and for generating synthetic SIFT-like data when the dataset is
not available.

Also handles:
  - Assigning random integer labels to every base vector.
  - Generating random filter ranges (min_label, max_label) for every query.

SIFT dataset home: http://corpus-texmex.irisa.fr/
Expected files after download:
  sift/sift_base.fvecs   – 1 000 000 x 128 float32 vectors
  sift/sift_query.fvecs  –    10 000 x 128 float32 vectors
  sift/sift_groundtruth.ivecs – ground-truth ANN (unfiltered); we recompute
                                ground-truth per filter range ourselves.
"""

import os
import time
import struct
import numpy as np
import h5py
from pathlib import Path



class Timer:
    def __init__(self, name):
        self.name = name
        self.t0 = None

    def __enter__(self):
        self.t0 = time.perf_counter()
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        time_taken = (time.perf_counter() - self.t0) * 1000
        print(f"{self.name}: {time_taken:.4f} ms")


# ---------------------------------------------------------------------------
# SIFT file format helpers
# ---------------------------------------------------------------------------

def _read_fvecs(path: str) -> np.ndarray:
    """Read a .fvecs file and return an (N, D) float32 array."""
    with h5py.File(path, "r") as f:
        base = np.asarray(f["train"], dtype=np.float32) / 255.0
        queries = np.asarray(f["test"], dtype=np.float32) / 255.0
    return base, queries


def load_sift(base_dir: str = "./data",
              max_base: int | None = None,
              max_query: int | None = None):
    """
    Load SIFT vectors from *base_dir*.

    Parameters
    ----------
    base_dir  : directory that contains sift-128-euclidean.hdf5
    max_base  : if set, only load the first max_base base vectors (for quick
                experiments without loading all 1 M vectors)
    max_query : if set, only load the first max_query queries

    Returns
    -------
    base_vecs   : (N, 128) float32
    query_vecs  : (Q, 128) float32
    """
    dataset_path  = os.path.join(base_dir, "sift-128-euclidean.hdf5")

    if not os.path.exists(dataset_path):
        raise FileNotFoundError(
            f"Cannot find {dataset_path}.\n"
            "Download SIFT1M from https://storage.googleapis.com/ann-datasets/ann-benchmarks/sift-128-euclidean.hdf5"
        )

    base_vecs, query_vecs  = _read_fvecs(dataset_path)

    if max_base  is not None: base_vecs  = base_vecs[:max_base]
    if max_query is not None: query_vecs = query_vecs[:max_query]

    print(f"[data] Loaded SIFT  base={base_vecs.shape}  query={query_vecs.shape}")
    return base_vecs, query_vecs


# ---------------------------------------------------------------------------
# Synthetic data (fallback / quick experiments)
# ---------------------------------------------------------------------------

def generate_synthetic_data(n_base: int = 50_000,
                             n_query: int = 1_000,
                             dim: int = 128,
                             seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate SIFT-like random Gaussian vectors.

    Parameters
    ----------
    n_base  : number of database vectors
    n_query : number of query vectors
    dim     : vector dimensionality (SIFT uses 128)
    seed    : random seed for reproducibility

    Returns
    -------
    base_vecs   : (n_base,  dim) float32
    query_vecs  : (n_query, dim) float32
    """
    rng = np.random.default_rng(seed)
    base_vecs  = rng.standard_normal((n_base,  dim)).astype(np.float32)
    query_vecs = rng.standard_normal((n_query, dim)).astype(np.float32)

    # L2-normalise to unit sphere so that pairwise distances lie in [0, 2].
    # This makes LSH bin_width directly interpretable (e.g. 0.4 ≈ 20% of max
    # distance) and keeps the synthetic data consistent with normalised SIFT.
    base_vecs  /= np.linalg.norm(base_vecs,  axis=1, keepdims=True) + 1e-8
    query_vecs /= np.linalg.norm(query_vecs, axis=1, keepdims=True) + 1e-8

    print(f"[data] Synthetic (L2-normalised)  base={base_vecs.shape}  query={query_vecs.shape}")
    return base_vecs, query_vecs


# ---------------------------------------------------------------------------
# Label / filter-range generation
# ---------------------------------------------------------------------------

def assign_labels(n: int,
                  n_labels: int = 100,
                  seed: int = 0) -> np.ndarray:
    """
    Assign a random integer label in [0, n_labels) to each of the n base
    vectors, drawn uniformly at random.

    Parameters
    ----------
    n        : number of base vectors
    n_labels : number of distinct label values
    seed     : random seed

    Returns
    -------
    labels : (n,) int32 array
    """
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, n_labels, size=n, dtype=np.int32)
    return labels


def generate_filter_ranges(n_queries: int,
                            n_labels: int = 100,
                            min_selectivity: float = 0.05,
                            max_selectivity: float = 0.40,
                            seed: int = 1) -> np.ndarray:
    """
    Generate a random filter range [lo, hi] for each query.

    The range width is drawn so that the fraction of labels covered
    (i.e. (hi - lo + 1) / n_labels) falls within
    [min_selectivity, max_selectivity].  This controls how selective the
    filters are – a key variable highlighted in the paper.

    Parameters
    ----------
    n_queries       : number of query vectors
    n_labels        : total number of distinct label values
    min_selectivity : minimum fraction of label space covered by a filter
    max_selectivity : maximum fraction of label space covered by a filter
    seed            : random seed

    Returns
    -------
    ranges : (n_queries, 2) int32  –  each row is [lo, hi]
    """
    rng = np.random.default_rng(seed)

    min_width = max(1, int(min_selectivity * n_labels))
    max_width = max(min_width, int(max_selectivity * n_labels))

    widths = rng.integers(min_width, max_width + 1, size=n_queries)
    lo     = rng.integers(0, n_labels, size=n_queries, dtype=np.int32)
    hi     = np.clip(lo + widths - 1, 0, n_labels - 1).astype(np.int32)

    # Ensure lo <= hi (should always hold, but be safe)
    lo, hi = np.minimum(lo, hi), np.maximum(lo, hi)
    return np.stack([lo, hi], axis=1)
