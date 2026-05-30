"""
prefilter.py
============
Pre-filtering approach to filtered vector search.

Algorithm (from the paper, Section 2.2):
  1. FILTER  – retain only those base vectors whose label falls in [lo, hi].
  2. KNN     – run exact K-nearest-neighbor search on the filtered subset.

This is the brute-force baseline: it is *exact* given the filter, so it also
serves as the ground-truth generator used by evaluate.py to compute recall
for the LSH post-filtering approach.

Time complexity per query: O(|S_f| · D)
  where |S_f| is the number of vectors passing the filter and D is dimension.

When to prefer pre-filtering (per the paper):
  - Filter is very selective  → |S_f| is small → KNN is cheap.
  - No vector index is available or its recall degrades badly.
"""

import time
import numpy as np

class Timer:
    def __init__(self, name):
        self.name = name
        self.t0 = None

    def __enter__(self):
        self.t0 = time.perf_counter()
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        time_taken = (time.perf_counter() - self.t0) * 1000
        print(f"{self.name}: {time_taken:.4f} ms")


class PreFilterSearch:
    """
    Exact filtered nearest-neighbor search via pre-filtering + brute-force KNN.

    Parameters
    ----------
    base_vecs : (N, D) float32  –  the database vectors
    labels    : (N,)   int32    –  per-vector integer attribute value
    """

    def __init__(self, base_vecs: np.ndarray, labels: np.ndarray):
        self.base_vecs = base_vecs          # (N, D)
        self.labels    = labels             # (N,)
        self.N, self.D = base_vecs.shape

    # ------------------------------------------------------------------
    # Core search
    # ------------------------------------------------------------------

    def search(self,
               query: np.ndarray,
               lo: int, hi: int,
               k: int = 10) -> np.ndarray:
        """
        Return the indices of the k nearest base vectors whose label is in
        [lo, hi], using exact L2 distance.

        Parameters
        ----------
        query : (D,) float32
        lo    : lower bound of the filter range (inclusive)
        hi    : upper bound of the filter range (inclusive)
        k     : number of neighbors to return

        Returns
        -------
        indices : (k',) int64  where k' = min(k, |filtered set|)
                  Indices into the *original* base_vecs array.
        """
        # Step 1 – filter
        mask             = (self.labels >= lo) & (self.labels <= hi)
        filtered_indices = np.where(mask)[0]

        if len(filtered_indices) == 0:
            return np.array([], dtype=np.int64)

        # Step 2 – exact KNN on filtered subset (vectorised L2)
        diff  = self.base_vecs[filtered_indices] - query   # (|S_f|, D)
        dists = np.einsum("nd,nd->n", diff, diff)          # squared L2; shape (|S_f|,)

        k_eff   = min(k, len(filtered_indices))
        top_idx = np.argpartition(dists, k_eff - 1)[:k_eff]
        top_idx = top_idx[np.argsort(dists[top_idx])]     # sort the top-k by distance

        return filtered_indices[top_idx]

    # ------------------------------------------------------------------
    # Batch search (used by evaluate.py)
    # ------------------------------------------------------------------

    def batch_search(self,
                     query_vecs: np.ndarray,
                     filter_ranges: np.ndarray,
                     k: int = 10) -> tuple[list[np.ndarray], float]:
        """
        Run search() for every query and record total wall-clock time.

        Parameters
        ----------
        query_vecs    : (Q, D) float32
        filter_ranges : (Q, 2) int32  –  columns are [lo, hi]
        k             : number of neighbors

        Returns
        -------
        results : list of Q arrays, each containing up to k indices
        elapsed : total search time in seconds (excludes any setup)
        """
        results = []
        t0 = time.perf_counter()
        for q_vec, (lo, hi) in zip(query_vecs, filter_ranges):
            results.append(self.search(q_vec, int(lo), int(hi), k))
        elapsed = time.perf_counter() - t0
        return results, elapsed
