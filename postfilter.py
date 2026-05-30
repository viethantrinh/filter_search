"""
postfilter.py
=============
Post-filtering approach to filtered vector search.

Algorithm (Section 2.2 of the paper):
  1. ANN SEARCH   – LSH retrieves a candidate set C from the full index.
  2. POST-FILTER  – discard candidates whose label falls outside [lo, hi].
  3. RERANK       – sort survivors by exact L2 distance; return top-k.

Optimisation vs the previous version:
  batch_search() now calls lsh.batch_query() which computes ALL Q query
  projections in a single (Q × D × L*K) matmul, then does Q×L dict lookups.
  The filter and rerank steps are also vectorised per query.
"""

import time
import numpy as np

from lsh_index import E2LSH_optimized


class PostFilterSearch:
    """
    Approximate filtered nearest-neighbor search via LSH + post-filtering.

    Parameters
    ----------
    base_vecs    : (N, D) float32
    labels       : (N,)   int32
    k_multiplier : (unused – kept for API compatibility; LSH returns all
                   candidates in the probed buckets automatically)
    **lsh_params : forwarded to E2LSH.__init__()
    """

    def __init__(self,
                 base_vecs: np.ndarray,
                 labels: np.ndarray,
                 k_multiplier: int = 5,   # kept for interface compatibility
                 **lsh_params):

        self.base_vecs    = base_vecs
        self.labels       = labels
        self.k_multiplier = k_multiplier
        self.N, self.D    = base_vecs.shape

        dim = lsh_params.pop("dim", self.D)
        self.lsh = E2LSH_optimized(dim=dim, **lsh_params)
        print(f"[PostFilter] Building LSH index over {self.N:,} vectors …")
        t0 = time.perf_counter()
        self.lsh.build(base_vecs, labels)
        print(f"[PostFilter] Index built in {time.perf_counter() - t0:.2f}s")

    # ------------------------------------------------------------------
    # Core single-query search
    # ------------------------------------------------------------------

    def search(self,
               query: np.ndarray,
               lo: int, hi: int,
               k: int = 10) -> np.ndarray:
        """
        Single-query filtered ANN search.
        For benchmarking full batches prefer batch_search().
        """
        candidates = self.lsh.query(query, lo, hi)
        return self._filter_and_rerank(query, candidates, lo, hi, k)

    # ------------------------------------------------------------------
    # Batch search  (optimised: one matmul for all queries)
    # ------------------------------------------------------------------

    def batch_search(self,
                     query_vecs: np.ndarray,
                     filter_ranges: np.ndarray,
                     k: int = 10) -> tuple[list[np.ndarray], float]:
        """
        Run filtered ANN search for all Q queries.

        The LSH projection step (the expensive part) is done in a single
        batched matrix multiply via lsh.batch_query().

        Parameters
        ----------
        query_vecs    : (Q, D) float32
        filter_ranges : (Q, 2) int32  –  columns [lo, hi]
        k             : number of neighbors

        Returns
        -------
        results : list of Q int32 arrays (up to k indices each)
        elapsed : search time in seconds (excludes index build)
        """
        t0 = time.perf_counter()

        # One batched matmul for all queries
        all_candidates = self.lsh.batch_query(query_vecs, filter_ranges)   # list of Q arrays

        results = []
        for q_vec, cands, (lo, hi) in zip(query_vecs, all_candidates, filter_ranges):
            results.append(
                self._filter_and_rerank(q_vec, cands, int(lo), int(hi), k))

        elapsed = time.perf_counter() - t0
        return results, elapsed

    # ------------------------------------------------------------------
    # Shared helper: post-filter + exact rerank
    # ------------------------------------------------------------------

    def _filter_and_rerank(self,
                           query: np.ndarray,
                           candidates: np.ndarray,
                           lo: int, hi: int,
                           k: int) -> np.ndarray:
        if len(candidates) == 0:
            return np.array([], dtype=np.int32)

        # Step 2 – label filter (vectorised boolean mask)
        lbl          = self.labels[candidates]
        valid        = candidates[(lbl >= lo) & (lbl <= hi)]
        # valid = candidates

        if len(valid) == 0:
            return np.array([], dtype=np.int32)

        # Step 3 – exact L2 rerank
        diff  = self.base_vecs[valid] - query
        dists = np.einsum("nd,nd->n", diff, diff)

        k_eff   = min(k, len(valid))
        top_idx = np.argpartition(dists, k_eff - 1)[:k_eff]
        top_idx = top_idx[np.argsort(dists[top_idx])]
        return valid[top_idx]

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def candidate_stats(self,
                        query_vecs: np.ndarray,
                        filter_ranges: np.ndarray) -> dict:
        all_cands = self.lsh.batch_query(query_vecs, filter_ranges)
        total, surviving = [], []
        for cands, (lo, hi) in zip(all_cands, filter_ranges):
            total.append(len(cands))
            if len(cands):
                mask = (self.labels[cands] >= lo) & (self.labels[cands] <= hi)
                surviving.append(int(mask.sum()))
            else:
                surviving.append(0)
        tc, sc = np.array(total), np.array(surviving)
        return {
            "avg_total_candidates"         : float(tc.mean()),
            "avg_surviving_candidates"     : float(sc.mean()),
            "avg_survival_rate"            : float((sc / np.maximum(tc, 1)).mean()),
            "queries_with_zero_candidates" : int((tc == 0).sum()),
            "queries_with_zero_surviving"  : int((sc == 0).sum()),
        }
