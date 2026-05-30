"""
postfilter.py
=============
Post-filtering approach to filtered vector search.

Algorithm (from the paper, Section 2.2):
  1. ANN SEARCH   – run LSH to retrieve a candidate set C from the full index.
  2. POST-FILTER  – discard candidates whose label falls *outside* [lo, hi].
  3. RERANK       – sort the surviving candidates by exact L2 distance and
                    return the top-k.

Key insight from the paper:
  The recall of post-filtering degrades when the filter is *selective* because
  many of the C candidates returned by ANN will be discarded, leaving fewer
  than k valid results.  A common heuristic is to inflate the ANN request to
  retrieve k_multiplier * k candidates, hoping enough survive the filter.
  We expose *k_multiplier* as a tunable parameter so you can study this
  recall–latency trade-off directly.

When to prefer post-filtering (per the paper):
  - Filter is *not* very selective (large fraction of base vectors pass).
  - The ANN index has good recall without the filter.

Design notes for this implementation:
  - The LSH index is built once on *all* base vectors (no label information).
  - At query time, the label filter is applied *after* ANN retrieval.
  - This means the LSH index itself is filter-agnostic (simpler to implement),
    matching the "generic index" scenario described in the paper.
"""

import time
import numpy as np

from lsh_index import E2LSH
from data_utils import Timer


class PostFilterSearch:
    """
    Approximate filtered nearest-neighbor search via LSH + post-filtering.

    Parameters
    ----------
    base_vecs    : (N, D) float32  –  the database vectors
    labels       : (N,)   int32    –  per-vector integer attribute value
    lsh_params   : keyword arguments forwarded to E2LSH.__init__()
                   e.g. n_tables=10, n_functions=4, bin_width=4.0
    k_multiplier : inflate the ANN request by this factor before filtering.
                   Larger values improve recall at the cost of more distance
                   computations.  The paper calls this an "extra tuning
                   parameter" compared to inline filtering.
    """

    def __init__(self,
                 base_vecs: np.ndarray,
                 labels: np.ndarray,
                 k_multiplier: int = 5,
                 **lsh_params):

        self.base_vecs    = base_vecs
        self.labels       = labels
        self.k_multiplier = k_multiplier
        self.N, self.D    = base_vecs.shape

        # Build the LSH index over *all* base vectors
        dim = lsh_params.pop("dim", self.D)
        self.lsh = E2LSH(dim=dim, **lsh_params)
        print(f"[PostFilter] Building LSH index over {self.N} vectors …")
        t0 = time.perf_counter()
        self.lsh.build(base_vecs)
        print(f"[PostFilter] Index built in {time.perf_counter() - t0:.2f}s")

    # ------------------------------------------------------------------
    # Core search
    # ------------------------------------------------------------------

    def search(self,
               query: np.ndarray,
               lo: int, hi: int,
               k: int = 10) -> np.ndarray:
        """
        Return the approximate k nearest base vectors whose label is in [lo, hi].

        Parameters
        ----------
        query : (D,) float32
        lo    : lower bound of the filter range (inclusive)
        hi    : upper bound of the filter range (inclusive)
        k     : number of neighbors to return

        Returns
        -------
        indices : (k',) int64  where k' ≤ k
                  Indices into the original base_vecs array.
        """
        # Step 1 – ANN: retrieve candidates from all LSH tables
        # with Timer("search"):
        candidates = self.lsh.query(query)

        if len(candidates) == 0:
            return np.array([], dtype=np.int64)

        # Step 2 – POST-FILTER: keep only those satisfying the label predicate
        # with Timer("filter"):
        label_mask      = (self.labels[candidates] >= lo) & \
                        (self.labels[candidates] <= hi)
        valid_candidates = candidates[label_mask]

        if len(valid_candidates) == 0:
            return np.array([], dtype=np.int64)

        # Step 3 – RERANK by exact L2 distance
        # with Timer("reranking"):
        diff  = self.base_vecs[valid_candidates] - query   # (C', D)
        dists = np.einsum("nd,nd->n", diff, diff)           # squared L2

        k_eff   = min(k, len(valid_candidates))
        top_idx = np.argpartition(dists, k_eff - 1)[:k_eff]
        top_idx = top_idx[np.argsort(dists[top_idx])]

        return valid_candidates[top_idx]

    # ------------------------------------------------------------------
    # Batch search
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
        elapsed : total search time in seconds (excludes index build time)
        """
        results = []
        t0 = time.perf_counter()
        for q_vec, (lo, hi) in zip(query_vecs, filter_ranges):
            results.append(self.search(q_vec, int(lo), int(hi), k))
        elapsed = time.perf_counter() - t0
        return results, elapsed

    # ------------------------------------------------------------------
    # Diagnostic: candidate set statistics
    # ------------------------------------------------------------------

    def candidate_stats(self,
                        query_vecs: np.ndarray,
                        filter_ranges: np.ndarray) -> dict:
        """
        Return statistics about the candidate sets (useful for ablation studies).

        Useful for understanding how filter selectivity affects post-filtering:
        a selective filter means many candidates are discarded in Step 2.
        """
        total_cands    = []
        surviving_cands = []

        for q_vec, (lo, hi) in zip(query_vecs, filter_ranges):
            cands = self.lsh.query(q_vec)
            total_cands.append(len(cands))
            if len(cands) > 0:
                mask = (self.labels[cands] >= lo) & (self.labels[cands] <= hi)
                surviving_cands.append(mask.sum())
            else:
                surviving_cands.append(0)

        tc = np.array(total_cands)
        sc = np.array(surviving_cands)
        return {
            "avg_total_candidates"     : float(tc.mean()),
            "avg_surviving_candidates" : float(sc.mean()),
            "avg_survival_rate"        : float((sc / np.maximum(tc, 1)).mean()),
            "queries_with_zero_candidates" : int((tc == 0).sum()),
            "queries_with_zero_surviving"  : int((sc == 0).sum()),
        }
