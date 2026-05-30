"""
ivf_index.py
============
A from-scratch IVF (Inverted File) index for filtered Euclidean ANN search,
designed to MAXIMISE the competition score:

        S = (QPS / 100) * Recall@50 ** 2

Why IVF instead of LSH for this score?
--------------------------------------
The score is dominated by QPS, not recall (recall only enters squared, but a
recall of 0.7 vs 1.0 is at most a 2x factor, whereas the LSH baseline leaves
~10x of QPS on the table). The LSH baseline is slow because the query path is a
Python loop doing L=400 dict lookups + a sort + a dedup *per query*.

IVF replaces that with:
  1. ONE matmul: (Q, D) @ (D, n_centroids)  -> pick nprobe nearest lists/query
  2. Gathering a small, bounded candidate set from those lists
  3. A vectorised label filter + exact L2 rerank

The candidate set per query is ~ nprobe * (N / n_centroids), which is tiny and
predictable, so reranking is cheap and recall is tunable via `nprobe`.

Filter-awareness
----------------
Because the label filter discards most candidates under selective filters, we
store, for every inverted list, its members SORTED BY LABEL. That lets us slice
out exactly the rows whose label is in [lo, hi] with np.searchsorted instead of
scanning + masking the whole list. This is the single biggest filtered-search
speedup.
"""

import time
import numpy as np


class IVFIndex:
    """
    Inverted-file index with label-sorted lists for fast range filtering.

    Parameters
    ----------
    n_centroids : number of coarse clusters (Voronoi cells). Rule of thumb:
                  ~ sqrt(N) to 4*sqrt(N). More centroids -> smaller lists ->
                  faster search but you must probe more of them for recall.
    n_iter      : k-means iterations for centroid training (a few is plenty).
    seed        : RNG seed for reproducible centroid init.
    """

    def __init__(self, n_centroids: int = 1024, n_iter: int = 10, seed: int = 42):
        self.n_centroids = n_centroids
        self.n_iter = n_iter
        self.seed = seed

        self.centroids: np.ndarray | None = None     # (C, D) float32
        self.base_vecs: np.ndarray | None = None      # (N, D) float32
        self.labels: np.ndarray | None = None         # (N,)   int32

        # Per-list data, all laid out contiguously for cache-friendly gathers:
        #   self.list_ptr[c] : self.list_ptr[c+1]  -> slice into the flat arrays
        self.list_ptr: np.ndarray | None = None       # (C+1,) int64
        self.member_ids: np.ndarray | None = None      # (N,) int32  global ids
        self.member_labels: np.ndarray | None = None   # (N,) int32  labels, sorted within list

    # ------------------------------------------------------------------
    # Training (k-means, NumPy)
    # ------------------------------------------------------------------

    def _train_kmeans(self, vecs: np.ndarray) -> np.ndarray:
        rng = np.random.default_rng(self.seed)
        N = len(vecs)
        C = min(self.n_centroids, N)

        # k-means++ style spread-out init is overkill; random sample is fine and fast.
        init_idx = rng.choice(N, size=C, replace=False)
        centroids = vecs[init_idx].copy()

        # Train on a subsample for speed if N is large (centroids barely change).
        train_n = min(N, max(50_000, C * 50))
        train = vecs if N <= train_n else vecs[rng.choice(N, train_n, replace=False)]

        for _ in range(self.n_iter):
            # Assign: argmin ||x - c||^2 = argmax (x.c - 0.5||c||^2)
            c_sq = np.einsum("cd,cd->c", centroids, centroids)        # (C,)
            scores = train @ centroids.T - 0.5 * c_sq                  # (n, C)
            assign = scores.argmax(axis=1)                             # (n,)

            # Update: mean of assigned points (vectorised via bincount per dim)
            new = np.zeros_like(centroids)
            counts = np.bincount(assign, minlength=C).astype(np.float32)
            np.add.at(new, assign, train)                              # scatter-add
            nonempty = counts > 0
            new[nonempty] /= counts[nonempty, None]
            # Re-seed empty clusters with random training points.
            if (~nonempty).any():
                reseed = rng.choice(len(train), (~nonempty).sum(), replace=False)
                new[~nonempty] = train[reseed]
            centroids = new.astype(np.float32)

        return centroids

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build(self, base_vecs: np.ndarray, labels: np.ndarray) -> None:
        self.base_vecs = np.ascontiguousarray(base_vecs, dtype=np.float32)
        self.labels = np.ascontiguousarray(labels, dtype=np.int32)
        N = len(base_vecs)

        t0 = time.perf_counter()
        self.centroids = self._train_kmeans(self.base_vecs)
        C = len(self.centroids)

        # Assign every base vector to its nearest centroid (batched, chunked).
        assign = np.empty(N, dtype=np.int32)
        c_sq = np.einsum("cd,cd->c", self.centroids, self.centroids)
        CHUNK = 100_000
        for s in range(0, N, CHUNK):
            e = min(s + CHUNK, N)
            sc = self.base_vecs[s:e] @ self.centroids.T - 0.5 * c_sq
            assign[s:e] = sc.argmax(axis=1)

        # Build CSR-style inverted lists, with members sorted by (list, label).
        # Sorting by a composite key groups by list and orders by label inside.
        order = np.lexsort((self.labels, assign))      # primary=assign, secondary=label
        sorted_assign = assign[order]
        self.member_ids = order.astype(np.int32)
        self.member_labels = self.labels[order]

        # CSR pointers
        counts = np.bincount(sorted_assign, minlength=C)
        self.list_ptr = np.zeros(C + 1, dtype=np.int64)
        np.cumsum(counts, out=self.list_ptr[1:])

        # ---- Global label-sorted view (for the exact pre-filter fallback) ----
        # When a filter is very selective, the surviving subset is tiny and an
        # EXACT brute-force KNN over it is both fast AND recall=1.0. We sort all
        # ids by label once so that, for any range [lo,hi], the surviving ids are
        # a single contiguous slice found via two binary searches.
        g_order = np.argsort(self.labels, kind="stable")
        self._global_ids_by_label = g_order.astype(np.int32)   # (N,)
        self._global_sorted_labels = self.labels[g_order]       # (N,)

        print(f"[IVF] Build done: {N:,} vectors | {C} lists | "
              f"avg list size = {N / C:.1f} | {time.perf_counter() - t0:.2f}s")

    # ------------------------------------------------------------------
    # Exact KNN over the filtered subset (used as a fallback for selective
    # filters). Fast because the surviving ids are one contiguous slice of the
    # globally label-sorted id array.
    # ------------------------------------------------------------------

    def _exact_filtered_knn(self, qv: np.ndarray, lo: int, hi: int,
                            k: int) -> np.ndarray:
        sl = self._global_sorted_labels
        a = np.searchsorted(sl, lo, side="left")
        b = np.searchsorted(sl, hi, side="right")
        if b <= a:
            return np.empty(0, dtype=np.int32)
        cand = self._global_ids_by_label[a:b]
        diff = self.base_vecs[cand] - qv
        dist = np.einsum("nd,nd->n", diff, diff)
        keff = min(k, len(cand))
        top = np.argpartition(dist, keff - 1)[:keff]
        top = top[np.argsort(dist[top])]
        return cand[top]

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def batch_search(self,
                     query_vecs: np.ndarray,
                     filter_ranges: np.ndarray,
                     k: int = 50,
                     nprobe: int = 16,
                     pre_filter_threshold: int = 0) -> tuple[list[np.ndarray], float]:
        """
        HYBRID filtered ANN search for all queries (recall-boosting version).

        Per-query strategy selection (the UNIFY/iRangeGraph idea):
          * If the filter is SELECTIVE (the surviving subset has <=
            `pre_filter_threshold` points), the IVF path would have very few
            candidates and low recall -> instead run an EXACT pre-filter KNN.
            The subset is small, so this is fast AND gives recall = 1.0.
          * Otherwise use the IVF path: probe the nprobe nearest lists, slice by
            label range (label-sorted lists -> searchsorted), rerank by L2.

        Set `pre_filter_threshold = 0` to disable the fallback (pure IVF).
        A good starting value is a few thousand (e.g. 2000-5000): big enough that
        exact KNN over the subset is still cheap, small enough to rescue the
        selective-filter queries where IVF recall collapses.

        Steps for the IVF path:
          1. ONE matmul to score queries against all centroids.
          2. Pick the nprobe nearest lists per query (argpartition).
          3. Gather members of those lists, slice by label range, rerank by L2.

        Returns (results, elapsed_seconds).
        """
        Q = len(query_vecs)
        query_vecs = np.ascontiguousarray(query_vecs, dtype=np.float32)
        lo_arr = filter_ranges[:, 0].astype(np.int32)
        hi_arr = filter_ranges[:, 1].astype(np.int32)

        t0 = time.perf_counter()

        # Pre-compute, for every query, how many base vectors survive the filter.
        # Two binary searches into the globally label-sorted labels -> O(log N).
        sl = self._global_sorted_labels
        lo_pos = np.searchsorted(sl, lo_arr, side="left")
        hi_pos = np.searchsorted(sl, hi_arr, side="right")
        survive = hi_pos - lo_pos                                   # (Q,) ints

        use_exact = (pre_filter_threshold > 0) & (survive <= pre_filter_threshold)

        # (1) Coarse assignment for all queries in one matmul.
        c_sq = np.einsum("cd,cd->c", self.centroids, self.centroids)
        sims = query_vecs @ self.centroids.T - 0.5 * c_sq             # (Q, C)
        # nprobe nearest centroids per query (unordered is fine).
        probe = np.argpartition(-sims, nprobe - 1, axis=1)[:, :nprobe]  # (Q, nprobe)

        base = self.base_vecs
        ids = self.member_ids
        lbls = self.member_labels
        ptr = self.list_ptr

        results: list[np.ndarray] = []
        for q in range(Q):
            qv = query_vecs[q]
            lo, hi = lo_arr[q], hi_arr[q]

            # ---- Exact fallback for selective filters (recall = 1.0) ----
            if use_exact[q]:
                results.append(self._exact_filtered_knn(qv, lo, hi, k))
                continue

            # ---- IVF path ----
            id_parts = []
            for c in probe[q]:
                s, e = ptr[c], ptr[c + 1]
                if e == s:
                    continue
                # Members of list c are sorted by label -> binary-search the range.
                lseg = lbls[s:e]
                a = s + np.searchsorted(lseg, lo, side="left")
                b = s + np.searchsorted(lseg, hi, side="right")
                if b > a:
                    id_parts.append(ids[a:b])

            if not id_parts:
                results.append(np.empty(0, dtype=np.int32))
                continue

            cand = np.concatenate(id_parts)
            # A vector can sit in only one list, so no dedup needed across probed
            # lists (Voronoi cells are disjoint). Rerank by exact L2.
            diff = base[cand] - qv
            dist = np.einsum("nd,nd->n", diff, diff)
            keff = min(k, len(cand))
            top = np.argpartition(dist, keff - 1)[:keff]
            top = top[np.argsort(dist[top])]
            results.append(cand[top])

        return results, time.perf_counter() - t0
