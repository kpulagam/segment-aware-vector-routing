#!/usr/bin/env python3
"""
Exp 15: Scale experiment (R1: "larger-scale experiments").

Runs the core Mean vs HMRC vs kRt comparison on growing corpus sizes
(default 100K -> 1M) with semi-structured segments, measuring routing
recall, build time, routing latency, and memory overhead at each scale.

Input: msmarco_1m.npy from prepare_revision_datasets.py --task scale
(or any large (n, d) float32 .npy).

Usage:
    python exp15_scale.py --embeddings data/revision/msmarco_1m.npy \
        --output revision_results/exp15 --sizes 100000 250000 500000 1000000 \
        --seeds 42 43 44
"""

import os
os.environ.setdefault('OMP_NUM_THREADS', '4')

import argparse
import time

import numpy as np
import pandas as pd

from common import (make_topic_labels, make_segments, sample_queries,
                    route_topk_segments, routing_recall, agg_mean_std)

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.hmrc import MeanCentroidIndex, HMRCIndex
from src.baselines import KRtStyleIndex

K_EVAL = [1, 5, 10]


def run_one(vectors, seed, size, segment_size=500, reps=(3,),
            with_krt=True):
    # topics scale with corpus so topic granularity stays comparable
    n_topics = max(50, size // 1000)
    labels = make_topic_labels(vectors, n_topics, seed=seed)
    assignments = make_segments(vectors, labels, 'semi', segment_size,
                                seed=seed)
    queries, gt = sample_queries(vectors, assignments, 2000, seed=seed)
    out = []
    methods = [('Mean', lambda: MeanCentroidIndex())]
    for r in reps:
        methods.append((f'HMRC-{r}',
                        lambda r=r: HMRCIndex(n_representatives=r)))
    if with_krt:
        methods.append(('kRt-b2d2',
                        lambda: KRtStyleIndex(branching=2, depth=2,
                                              seed=seed)))
    for name, ctor in methods:
        idx = ctor()
        t0 = time.time()
        idx.fit(vectors, assignments)
        build_t = time.time() - t0
        reps = getattr(idx, 'all_representatives', None)
        r2s = idx.rep_to_segment if reps is not None else idx.segment_ids
        if reps is None:
            reps = idx.centroids
        t0 = time.time()
        routed = route_topk_segments(queries, reps, r2s, max(K_EVAL))
        route_us = (time.time() - t0) / len(queries) * 1e6
        row = {'size': size, 'seed': seed, 'method': name,
               'segment_size': segment_size,
               'n_segments': len(np.unique(assignments)),
               'n_reps': len(reps), 'build_time_s': build_t,
               'route_us_per_query': route_us,
               'centroid_mem_mb': reps.nbytes / 1e6}
        for k in K_EVAL:
            row[f'recall@{k}'] = routing_recall(routed, gt, k)
        out.append(row)
        print(f"[size={size} seed={seed}] {name}: "
              f"R@10={row['recall@10']:.3f} build={build_t:.1f}s "
              f"route={route_us:.0f}us", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--embeddings', required=True)
    ap.add_argument('--output', default='revision_results/exp15')
    ap.add_argument('--sizes', type=int, nargs='+',
                    default=[100000, 250000, 500000, 1000000])
    ap.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44])
    ap.add_argument('--segment-sizes', type=int, nargs='+', default=[500],
                    help='sweep production-like segment sizes, e.g. '
                         '500 2000 10000 50000')
    ap.add_argument('--reps', type=int, nargs='+', default=[3],
                    help='HMRC representative counts to evaluate, '
                         'e.g. 3 10 30')
    ap.add_argument('--no-krt', action='store_true',
                    help='skip the kRt baseline')
    args = ap.parse_args()

    full = np.load(args.embeddings, mmap_mode='r')
    os.makedirs(args.output, exist_ok=True)
    raw_path = os.path.join(args.output, 'exp15_raw.csv')

    rows = []
    for size in args.sizes:
        if size > len(full):
            print(f'skip size {size} > corpus {len(full)}')
            continue
        for seed in args.seeds:
            rng = np.random.RandomState(seed)
            sel = np.sort(rng.choice(len(full), size, replace=False))
            vectors = np.ascontiguousarray(full[sel]).astype(np.float32)
            for seg_size in args.segment_sizes:
                rows.extend(run_one(vectors, seed, size, seg_size,
                                    reps=tuple(args.reps),
                                    with_krt=not args.no_krt))
            del vectors
            df = pd.DataFrame(rows)
            if os.path.exists(raw_path):
                prev = pd.read_csv(raw_path)
                if 'segment_size' not in prev.columns:
                    prev['segment_size'] = 500
                df = pd.concat([prev, df], ignore_index=True).drop_duplicates(
                    subset=['size', 'seed', 'method', 'segment_size'],
                    keep='last')
            df.to_csv(raw_path, index=False)
            rows = []

    df = pd.read_csv(raw_path)
    if 'segment_size' not in df.columns:
        df['segment_size'] = 500
    agg = agg_mean_std(df, ['size', 'segment_size', 'method'],
                       [f'recall@{k}' for k in K_EVAL] +
                       ['build_time_s', 'route_us_per_query',
                        'centroid_mem_mb'])
    agg.to_csv(os.path.join(args.output, 'exp15_aggregated.csv'),
               index=False)
    print(agg.to_string(index=False))


if __name__ == '__main__':
    main()
