"""
sweep_ivf.py
============
Stand-alone benchmark + parameter sweep for the IVF filtered-ANN method.

It loads SIFT, builds the exact ground truth once (pre-filter), then sweeps
IVF parameters (n_centroids x nprobe) and prints Recall@50, QPS and the
competition score:

        S = (QPS / 100) * Recall@50 ** 2

The row with the highest score is highlighted at the end so you can copy those
parameters straight into main.py.

Usage
-----
    # Full SIFT-1M (needs ~4-6 GB RAM for the exact ground-truth pass):
    python sweep_ivf.py

    # Smaller / faster smoke test on a subset:
    python sweep_ivf.py --max-base 200000 --n-query 2000

    # Custom grid:
    python sweep_ivf.py --centroids 2048 4096 8192 --nprobe 8 16 24 32 48
"""

import argparse
import time
import h5py
import numpy as np

from prefilter import PreFilterSearch
from ivf_search import IVFFilteredSearch
from data_utils import assign_labels, generate_filter_ranges
from evaluate import compute_recall, compute_qps


def parse_args():
    p = argparse.ArgumentParser(description="IVF filtered-ANN parameter sweep")
    p.add_argument("--sift-dir", default="./data",
                   help="dir containing sift-128-euclidean.hdf5")
    p.add_argument("--max-base", type=int, default=None,
                   help="limit base vectors (None = all 1M)")
    p.add_argument("--n-query", type=int, default=None,
                   help="limit queries (None = all 10K)")
    p.add_argument("--k", type=int, default=50)
    p.add_argument("--n-labels", type=int, default=1000)
    p.add_argument("--min-sel", type=float, default=0.05)
    p.add_argument("--max-sel", type=float, default=0.40)
    p.add_argument("--centroids", type=int, nargs="+", default=[4096])
    p.add_argument("--nprobe", type=int, nargs="+", default=[8, 16, 24, 32, 48])
    p.add_argument("--n-iter", type=int, default=10, help="k-means iterations")
    p.add_argument("--pre-thr", type=int, nargs="+", default=[0],
                   help="pre_filter_threshold values to sweep. If the filtered "
                        "subset has <= this many points, fall back to EXACT KNN "
                        "(recall=1.0). 0 disables. Try 0 2000 5000 10000.")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def load_sift(sift_dir, max_base, max_query):
    path = f"{sift_dir}/sift-128-euclidean.hdf5"
    with h5py.File(path, "r") as f:
        b = f["train"]
        q = f["test"]
        nb = b.shape[0] if max_base is None else min(max_base, b.shape[0])
        nq = q.shape[0] if max_query is None else min(max_query, q.shape[0])
        # slice on disk first -> only load what we need (memory friendly)
        base = np.asarray(b[:nb], dtype=np.float32) / 255.0
        query = np.asarray(q[:nq], dtype=np.float32) / 255.0
    print(f"[data] base={base.shape}  query={query.shape}")
    return base, query


def main():
    args = parse_args()

    base, query = load_sift(args.sift_dir, args.max_base, args.n_query)
    N, Q = len(base), len(query)

    labels = assign_labels(N, n_labels=args.n_labels, seed=args.seed)
    fr = generate_filter_ranges(Q, n_labels=args.n_labels,
                                min_selectivity=args.min_sel,
                                max_selectivity=args.max_sel,
                                seed=args.seed + 1)
    sel = (fr[:, 1] - fr[:, 0] + 1) / args.n_labels
    print(f"[config] N={N:,} Q={Q:,} k={args.k} "
          f"selectivity mean={sel.mean():.2%} "
          f"[{sel.min():.2%}, {sel.max():.2%}]")

    # ------------------------------------------------------------------
    # Ground truth (exact pre-filter) — computed ONCE.
    # ------------------------------------------------------------------
    print("\n[GT] computing exact ground truth (pre-filter) …")
    t0 = time.perf_counter()
    gt, t_pre = PreFilterSearch(base, labels).batch_search(query, fr, k=args.k)
    qps_pre = compute_qps(Q, t_pre)
    print(f"[GT] done in {time.perf_counter()-t0:.1f}s | "
          f"PreFilter QPS={qps_pre:.1f}  score={(qps_pre/100)*1.0:.3f}")

    # ------------------------------------------------------------------
    # Sweep
    # ------------------------------------------------------------------
    print("\n" + "=" * 74)
    print(f"  {'centroids':>9} {'nprobe':>7} {'pre_thr':>8} "
          f"{'Recall@K':>9} {'QPS':>10} {'SCORE':>9}")
    print("=" * 74)

    best = None
    for C in args.centroids:
        # Build the index once per centroid count, reuse across nprobe/threshold.
        searcher = IVFFilteredSearch(base, labels, n_centroids=C,
                                     nprobe=max(args.nprobe), n_iter=args.n_iter,
                                     seed=args.seed)
        for nprobe in sorted(args.nprobe):
            searcher.nprobe = nprobe
            for thr in sorted(args.pre_thr):
                searcher.pre_filter_threshold = thr
                res, t = searcher.batch_search(query, fr, k=args.k)
                R = compute_recall(res, gt)["mean_recall"]
                QPS = compute_qps(Q, t)
                S = (QPS / 100) * (R ** 2)
                marker = ""
                if best is None or S > best[0]:
                    best = (S, C, nprobe, thr, R, QPS)
                    marker = "  <- best so far"
                print(f"  {C:>9} {nprobe:>7} {thr:>8} "
                      f"{R:>9.4f} {QPS:>10.1f} {S:>9.4f}{marker}")

    print("=" * 74)
    S, C, nprobe, thr, R, QPS = best
    print(f"\nBEST: n_centroids={C}  nprobe={nprobe}  pre_filter_threshold={thr}  "
          f"Recall@{args.k}={R:.4f}  QPS={QPS:.1f}  SCORE={S:.4f}")
    print("\nPlug into main.py:")
    print(f"    post = PostFilterSearch(base_vecs, labels, "
          f"n_centroids={C}, nprobe={nprobe}, pre_filter_threshold={thr})")


if __name__ == "__main__":
    main()
