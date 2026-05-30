"""
ivf_search.py
=============
Drop-in replacement for PostFilterSearch that uses the IVF index.

Same constructor / batch_search() contract as postfilter.PostFilterSearch, so
you can swap it into main.py with a one-line import change. Designed to
maximise S = (QPS/100) * Recall@50**2 by giving far higher QPS than the LSH
post-filter at comparable recall.

Quick use in main.py
---------------------
    from ivf_search import IVFFilteredSearch as PostFilterSearch
    ...
    post = PostFilterSearch(base_vecs, labels,
                            n_centroids=1024, nprobe=24)
    post_results, time_post = post.batch_search(query_vecs, filter_ranges, k=args.k)

Tuning the recall/speed trade-off (and therefore the score):
    n_centroids : sqrt(N) .. 4*sqrt(N).  For N=1e6 try 2048-8192.
    nprobe      : how many lists to scan per query. Higher -> higher recall,
                  lower QPS. This is THE knob to sweep for the score.
"""

import time
import numpy as np

from ivf_index import IVFIndex


class IVFFilteredSearch:
    def __init__(self,
                 base_vecs: np.ndarray,
                 labels: np.ndarray,
                 n_centroids: int = 1024,
                 nprobe: int = 16,
                 pre_filter_threshold: int = 0,
                 n_iter: int = 10,
                 seed: int = 42,
                 # swallow LSH-style kwargs so it's a true drop-in:
                 **_ignored):
        self.base_vecs = base_vecs
        self.labels = labels
        self.nprobe = nprobe
        self.pre_filter_threshold = pre_filter_threshold
        self.N, self.D = base_vecs.shape

        self.index = IVFIndex(n_centroids=n_centroids, n_iter=n_iter, seed=seed)
        print(f"[IVFFilteredSearch] Building IVF index over {self.N:,} vectors …")
        t0 = time.perf_counter()
        self.index.build(base_vecs, labels)
        print(f"[IVFFilteredSearch] Index built in {time.perf_counter() - t0:.2f}s")

    def batch_search(self, query_vecs, filter_ranges, k: int = 50):
        return self.index.batch_search(query_vecs, filter_ranges, k=k,
                                        nprobe=self.nprobe,
                                        pre_filter_threshold=self.pre_filter_threshold)

    # single-query convenience (not used in the main benchmark loop)
    def search(self, query, lo, hi, k: int = 50):
        res, _ = self.index.batch_search(query[None, :],
                                         np.array([[lo, hi]]), k=k,
                                         nprobe=self.nprobe)
        return res[0]

    def candidate_stats(self, query_vecs, filter_ranges):
        # Lightweight stats for parity with PostFilterSearch's diagnostics.
        res, _ = self.index.batch_search(query_vecs, filter_ranges, k=10_000,
                                         nprobe=self.nprobe)
        sizes = np.array([len(r) for r in res])
        return {
            "avg_surviving_candidates": float(sizes.mean()),
            "queries_with_zero_surviving": int((sizes == 0).sum()),
        }
