#!/usr/bin/env python3
"""
Exp 12: Head-to-head comparison of HMRC (flat k-means representatives)
vs kRt-style sub-cluster routing (hierarchical k-means representatives),
at equal representative budgets, with multi-seed variance.

Addresses:
  R1: novelty positioning vs sub-cluster routing must be backed empirically.
  R2: "soften uniqueness claims around kRt/Pyramid unless experimentally
       compared" + "add confidence intervals or repeated-run variance".

Usage:
    python exp12_krt_comparison.py --embeddings data/embeddings/msmarco_100000.npy \
        --dataset-name MSMARCO-100K --output results_exp12 --seeds 42 43 44 45 46
"""

import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

import argparse
import json
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd

from common import (make_topic_labels, make_segments, sample_queries,
                    route_topk_segments, routing_recall, agg_mean_std)

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.hmrc import MeanCentroidIndex, HMRCIndex
from src.baselines import KRtStyleIndex

K_EVAL = [1, 5, 10]


def build_methods(seed):
    """(name, family, budget, constructor) — equal-budget pairs."""
    methods = [('Mean', 'mean', 1, lambda: MeanCentroidIndex())]
    for r in [3, 4, 9, 16]:
        methods.append((f'HMRC-{r}', 'flat', r,
                        lambda r=r: HMRCIndex(n_representatives=r)))
    for b, d in [(2, 2), (3, 2), (4, 2)]:
        methods.append((f'kRt-b{b}d{d}', 'hier', b ** d,
                        lambda b=b, d=d: KRtStyleIndex(branching=b, depth=d,
                                                       seed=seed)))
    return methods


def run_one(args):
    vectors, strategy, seed, cfg = args
    topic_labels = make_topic_labels(vectors, cfg['n_topics'], seed=seed)
    assignments = make_segments(vectors, topic_labels, strategy,
                                cfg['segment_size'], seed=seed)
    queries, gt = sample_queries(vectors, assignments, cfg['n_queries'],
                                 seed=seed)
    rows = []
    for name, family, budget, ctor in build_methods(seed):
        idx = ctor()
        t0 = time.time()
        idx.fit(vectors, assignments)
        build_t = time.time() - t0
        reps = getattr(idx, 'all_representatives', None)
        if reps is None:
            reps = idx.centroids
            rep_to_seg = idx.segment_ids
        else:
            rep_to_seg = idx.rep_to_segment
        t0 = time.time()
        routed = route_topk_segments(queries, reps, rep_to_seg, max(K_EVAL))
        route_t = (time.time() - t0) / len(queries) * 1e6  # us/query
        row = {
            'strategy': strategy, 'seed': seed, 'method': name,
            'family': family, 'nominal_budget': budget,
            'n_reps_total': len(reps),
            'avg_reps_per_seg': len(reps) / idx.n_segments
            if hasattr(idx, 'n_segments') and idx.n_segments else
            len(reps) / len(np.unique(assignments)),
            'build_time_s': build_t, 'route_us_per_query': route_t,
        }
        for k in K_EVAL:
            row[f'recall@{k}'] = routing_recall(routed, gt, k)
        rows.append(row)
        print(f"[{strategy} seed={seed}] {name}: "
              f"R@10={row['recall@10']:.3f} reps/seg={row['avg_reps_per_seg']:.1f}",
              flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--embeddings', required=True)
    ap.add_argument('--dataset-name', default='MSMARCO-100K')
    ap.add_argument('--output', default='results_exp12')
    ap.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44, 45, 46])
    ap.add_argument('--strategies', nargs='+', default=['semi', 'random'])
    ap.add_argument('--n-topics', type=int, default=100)
    ap.add_argument('--segment-size', type=int, default=500)
    ap.add_argument('--n-queries', type=int, default=2000)
    ap.add_argument('--workers', type=int, default=4)
    args = ap.parse_args()

    vectors = np.load(args.embeddings).astype(np.float32)
    cfg = {'n_topics': args.n_topics, 'segment_size': args.segment_size,
           'n_queries': args.n_queries}

    jobs = [(vectors, st, sd, cfg) for st in args.strategies
            for sd in args.seeds]
    if args.workers > 1:
        with Pool(args.workers) as pool:
            results = pool.map(run_one, jobs)
    else:
        results = [run_one(j) for j in jobs]

    df = pd.DataFrame([r for rows in results for r in rows])
    out = f"{args.output}_{args.dataset_name}"
    os.makedirs(out, exist_ok=True)
    raw_path = os.path.join(out, 'exp12_raw.csv')
    if os.path.exists(raw_path):
        prev = pd.read_csv(raw_path)
        df = pd.concat([prev, df], ignore_index=True)
        df = df.drop_duplicates(subset=['strategy', 'seed', 'method'],
                                keep='last')
    df.to_csv(raw_path, index=False)

    value_cols = [f'recall@{k}' for k in K_EVAL] + [
        'avg_reps_per_seg', 'build_time_s', 'route_us_per_query']
    agg = agg_mean_std(df, ['strategy', 'method', 'family',
                            'nominal_budget'], value_cols)
    agg = agg.sort_values(['strategy', 'nominal_budget', 'family'])
    agg.to_csv(os.path.join(out, 'exp12_aggregated.csv'), index=False)
    print(agg.to_string(index=False))
    with open(os.path.join(out, 'exp12_config.json'), 'w') as f:
        json.dump({**cfg, 'seeds': args.seeds,
                   'strategies': args.strategies,
                   'dataset': args.dataset_name}, f, indent=2)


if __name__ == '__main__':
    main()
