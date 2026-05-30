"""
evaluate.py
===========
Evaluation metrics for filtered approximate nearest-neighbor search.

Metrics
-------
Recall@K
    The standard ANN evaluation metric.  Given ground-truth G (the exact
    top-k filtered neighbors) and the ANN result R (up to k indices):

        Recall@K = |R ∩ G| / |G|

    Averaged over all queries.  Note: if the filter leaves fewer than k
    vectors the ground-truth set is smaller than k, but the denominator
    is still |G|, so perfect recall is still achievable.

QPS (Queries Per Second)
    Total queries / total search time (wall-clock, single thread).
    Index build time is excluded – we measure *search* throughput only.

Usage
-----
    from evaluate import compute_recall, compute_qps, print_comparison
"""

import numpy as np
from typing import Sequence

import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Recall@K
# ---------------------------------------------------------------------------

def compute_recall(
        results    : Sequence[np.ndarray],
        groundtruth: Sequence[np.ndarray],
) -> dict:
    """
    Compute Recall@K statistics across all queries.

    Parameters
    ----------
    results     : list of Q arrays – ANN result for each query
    groundtruth : list of Q arrays – exact KNN result for each query
                  (produced by PreFilterSearch)

    Returns
    -------
    dict with keys:
      "mean_recall"   : average recall over all queries
      "median_recall" : median recall
      "min_recall"    : worst-case recall (important for SLA-style guarantees)
      "max_recall"    : best-case recall
      "per_query"     : (Q,) float32 array of per-query recalls
    """
    assert len(results) == len(groundtruth), \
        "results and groundtruth must have the same number of queries"

    recalls = []
    for res, gt in zip(results, groundtruth):
        if len(gt) == 0:
            # No vectors satisfy the filter: recall is undefined; skip.
            continue
        # Intersection size (order-insensitive – recall is set-based)
        hits = len(np.intersect1d(res, gt))
        recalls.append(hits / len(gt))

    arr = np.array(recalls, dtype=np.float32)
    return {
        "mean_recall"   : float(arr.mean())   if len(arr) else 0.0,
        "median_recall" : float(np.median(arr)) if len(arr) else 0.0,
        "min_recall"    : float(arr.min())    if len(arr) else 0.0,
        "max_recall"    : float(arr.max())    if len(arr) else 0.0,
        "per_query"     : arr,
    }


# ---------------------------------------------------------------------------
# QPS
# ---------------------------------------------------------------------------

def compute_qps(n_queries: int, elapsed_seconds: float) -> float:
    """
    Queries Per Second.

    Parameters
    ----------
    n_queries      : total number of queries processed
    elapsed_seconds: wall-clock time for all queries (build time excluded)

    Returns
    -------
    qps : float
    """
    return n_queries / elapsed_seconds if elapsed_seconds > 0 else float("inf")


# ---------------------------------------------------------------------------
# Selectivity breakdown (extra analysis)
# ---------------------------------------------------------------------------

def compute_recall_by_selectivity(
        results       : Sequence[np.ndarray],
        groundtruth   : Sequence[np.ndarray],
        filter_ranges : np.ndarray,
        n_base        : int,
        n_labels      : int,
        n_bins        : int = 4,
) -> list[dict]:
    """
    Break down recall by filter selectivity bin.

    Selectivity = (hi - lo + 1) / n_labels  (fraction of label space covered).
    This lets you see how recall degrades as filters become more selective –
    a key phenomenon described in the paper.

    Parameters
    ----------
    results, groundtruth : as in compute_recall()
    filter_ranges : (Q, 2) int32
    n_base        : number of base vectors
    n_labels      : number of distinct label values
    n_bins        : number of selectivity bins

    Returns
    -------
    list of dicts, one per bin, each containing:
      "bin_label"     : human-readable range string e.g. "[0.00, 0.25)"
      "n_queries"     : queries in this bin
      "mean_recall"   : mean recall for this bin
      "mean_gt_size"  : average ground-truth set size
    """
    # Selectivity per query: fraction of the *label* space covered
    lo, hi       = filter_ranges[:, 0], filter_ranges[:, 1]
    selectivities = (hi - lo + 1).astype(float) / n_labels  # shape (Q,)

    bins    = np.linspace(0, 1, n_bins + 1)
    bin_ids = np.digitize(selectivities, bins[1:], right=False)  # 0 … n_bins-1

    rows = []
    for b in range(n_bins):
        idx = np.where(bin_ids == b)[0]
        if len(idx) == 0:
            continue
        bin_results = [results[i]     for i in idx]
        bin_gt      = [groundtruth[i] for i in idx]
        rec_dict    = compute_recall(bin_results, bin_gt)
        mean_gt     = np.mean([len(groundtruth[i]) for i in idx])
        lo_b = bins[b]
        hi_b = bins[b + 1]
        rows.append({
            "bin_label"    : f"[{lo_b:.2f}, {hi_b:.2f})",
            "n_queries"    : len(idx),
            "mean_recall"  : rec_dict["mean_recall"],
            "mean_gt_size" : float(mean_gt),
        })
    return rows


# ---------------------------------------------------------------------------
# Pretty-print comparison table
# ---------------------------------------------------------------------------

def plot_score_with_point(qps, recall, qps_max=200, levels=20):
    qps_upper = max(qps_max, qps * 1.1 if qps > 0 else qps_max)

    recall_grid = np.linspace(0, 1, 300)
    qps_grid = np.linspace(0, qps_upper, 300)
    R, Q = np.meshgrid(recall_grid, qps_grid)

    S = (Q / 100) * (R ** 2)

    s_point = (qps / 100) * (recall ** 2)

    fig, ax = plt.subplots(figsize=(8, 6))
    contour = ax.contourf(R, Q, S, levels=levels)
    plt.colorbar(contour, ax=ax, label='S')

    ax.scatter(recall, qps, s=120, c="red", marker='x', linewidths=2, label='Input point')
    ax.annotate(
        f'Recall={recall:.3f}\nQPS={qps:.1f}\nS={s_point:.4f}',
        xy=(recall, qps),
        xytext=(10, 10),
        textcoords='offset points'
    )

    ax.set_xlabel('Recall')
    ax.set_ylabel('QPS')
    ax.set_title(r'$S=\frac{QPS}{100}\cdot R^2$')
    plt.tight_layout()
    plt.grid(True)

    plt.savefig("./final_score.png", dpi=200, bbox_inches='tight')

def print_comparison(
        name_a: str, results_a: Sequence[np.ndarray], time_a: float,
        name_b: str, results_b: Sequence[np.ndarray], time_b: float,
        groundtruth: Sequence[np.ndarray],
        k: int,
        filter_ranges: np.ndarray | None = None,
        n_base: int | None = None,
        n_labels: int | None = None,
) -> None:
    """
    Print a side-by-side evaluation table for two methods.

    Parameters
    ----------
    name_a / name_b    : display names for the two methods
    results_a / _b     : search results from each method
    time_a / time_b    : search elapsed time (seconds) for each method
    groundtruth        : exact results (from PreFilterSearch)
    k                  : the K in Recall@K
    filter_ranges      : optional; if provided, print selectivity breakdown
    n_base             : total base vectors (for selectivity analysis)
    n_labels           : number of distinct labels (for selectivity analysis)
    """
    Q = len(groundtruth)

    rec_a = compute_recall(results_a, groundtruth)
    rec_b = compute_recall(results_b, groundtruth)
    qps_a = compute_qps(Q, time_a)
    qps_b = compute_qps(Q, time_b)

    get_score = lambda recall, qps: (qps / 100) * (recall ** 2)
    score_a = get_score(rec_a["mean_recall"], qps_a)
    score_b = get_score(rec_b["mean_recall"], qps_b)

    plot_score_with_point(qps=qps_b, recall=rec_b["mean_recall"])

    W = 60
    print()
    print("=" * W)
    print(f"  Filtered ANN Evaluation   (K={k}, Q={Q} queries)")
    print("=" * W)
    header = f"{'Metric':<28} {name_a:>13} {name_b:>13}"
    print(header)
    print("-" * W)

    def row(label, va, vb, fmt=".4f"):
        print(f"  {label:<26} {va:{'>13.' + fmt[1:]}} {vb:{'>13.' + fmt[1:]}}")

    row("Recall@K (mean)",    rec_a["mean_recall"],   rec_b["mean_recall"])
    row("Recall@K (median)",  rec_a["median_recall"], rec_b["median_recall"])
    row("Recall@K (min)",     rec_a["min_recall"],    rec_b["min_recall"])
    row("QPS",                qps_a,                  qps_b,   fmt=".1f")
    row("Final score",        score_a,                score_b, fmt=".2f")
    row("Search time (s)",    time_a,                 time_b,  fmt=".3f")
    print("=" * W)

    # Per-selectivity breakdown
    if filter_ranges is not None and n_base is not None and n_labels is not None:
        print()
        print("  Recall by filter selectivity (lower = more selective filter)")
        print("-" * W)
        bins_a = compute_recall_by_selectivity(
            results_a, groundtruth, filter_ranges, n_base, n_labels)
        bins_b = compute_recall_by_selectivity(
            results_b, groundtruth, filter_ranges, n_base, n_labels)

        print(f"  {'Selectivity bin':<18} {'n_q':>5} "
              f"{'Recall_A':>10} {'Recall_B':>10} {'avg |GT|':>10}")
        print("-" * W)
        for ba, bb in zip(bins_a, bins_b):
            print(f"  {ba['bin_label']:<18} {ba['n_queries']:>5} "
                  f"{ba['mean_recall']:>10.4f} {bb['mean_recall']:>10.4f} "
                  f"{ba['mean_gt_size']:>10.1f}")
        print("=" * W)
    print()
