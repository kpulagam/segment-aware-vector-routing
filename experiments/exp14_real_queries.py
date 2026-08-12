#!/usr/bin/env python3
"""
Exp 14: Evaluation with REAL query distributions (R2: "discuss how HMRC
behaves with real query distributions, not only held-out corpus vectors").

Consumes bundles produced by prepare_revision_datasets.py (doc embeddings,
real test-query embeddings, qrels). Reports:

  1. Routing recall@k with multi-ground-truth: a query is routed correctly
     if ANY segment containing a relevant document is probed.
  2. End-to-end retrieval hit@10 under IVF-style search: probe nprobe
     segments, exact search within, check whether a relevant doc is in the
     top-10 results. (Same protocol as the FAISS experiment, numpy-exact.)
  3. Side-by-side "pseudo query" protocol (held-out corpus vectors) on the
     same segmentation, so real vs synthetic query effects are directly
     comparable.

Usage:
    python exp14_real_queries.py --bundle data/revision/nq100k_v2.npz \
        --dataset-name NQ-100K-v2 --output revision_results/exp14 \
        --seeds 42 43 44 --strategies semi timebatch domain
"""

import os
os.environ.setdefault('OMP_NUM_THREADS', '4')

import argparse
import json
import time

import numpy as np
import pandas as pd

from common import (make_topic_labels, make_segments, pairwise_sq_dists,
                    route_topk_segments, routing_recall,
                    routing_recall_multi_gt, agg_mean_std)

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.hmrc import MeanCentroidIndex, HMRCIndex
from src.baselines import KRtStyleIndex

K_EVAL = [1, 5, 10, 20]
NPROBE_E2E = [1, 5, 10]
E2E_MAX_QUERIES = 2000


def load_bundle(path):
    z = np.load(path, allow_pickle=False)
    b = {k: z[k] for k in z.files}
    id_to_row = {d: i for i, d in enumerate(b['doc_ids'])}
    n_q = len(b['query_embs'])
    gt_rows = [[] for _ in range(n_q)]
    for qi, did in zip(b['qrel_query_idx'], b['qrel_doc_id']):
        r = id_to_row.get(did)
        if r is not None:
            gt_rows[int(qi)].append(r)
    b['gt_rows'] = gt_rows
    return b


def build_index(name, seed):
    if name == 'Mean':
        return MeanCentroidIndex()
    if name.startswith('HMRC-'):
        return HMRCIndex(n_representatives=int(name.split('-')[1]))
    if name == 'kRt-b2d2':
        return KRtStyleIndex(branching=2, depth=2, seed=seed)
    raise ValueError(name)


def rep_arrays(idx):
    reps = getattr(idx, 'all_representatives', None)
    if reps is None:
        return idx.centroids, idx.segment_ids
    return reps, idx.rep_to_segment


def e2e_hit_at(vectors, assignments, seg_rows, routed, queries, gt_rows,
               topn=10):
    """Probe routed segments, exact search inside, hit@topn per query."""
    hits, denom = 0, 0
    for i in range(len(queries)):
        gts = gt_rows[i]
        if not gts:
            continue
        denom += 1
        rows = np.concatenate([seg_rows[s] for s in routed[i] if s >= 0])
        d = pairwise_sq_dists(queries[i:i + 1], vectors[rows])[0]
        top = rows[np.argsort(d)[:topn]]
        if set(top) & set(gts):
            hits += 1
    return hits / max(denom, 1)


def run_one(bundle, strategy, seed, cfg):
    V = bundle['doc_embs'].astype(np.float32)
    if strategy == 'domain':
        labels = bundle['domains'].astype(int)
    else:
        labels = make_topic_labels(V, cfg['n_topics'], seed=seed)
    if strategy == 'domain':
        from src.segmentation import create_semi_structured_segments
        assignments = create_semi_structured_segments(
            V, labels, cfg['segment_size'],
            topics_per_segment=min(3, len(np.unique(labels))), seed=seed)
    else:
        assignments = make_segments(V, labels, strategy,
                                    cfg['segment_size'], seed=seed)
    seg_ids = np.unique(assignments)
    seg_rows = {s: np.where(assignments == s)[0] for s in seg_ids}

    queries = bundle['query_embs'].astype(np.float32)
    gt_rows = bundle['gt_rows']
    gt_seg_sets = [set(assignments[rows]) for rows in gt_rows]

    # pseudo-query protocol on same segmentation for comparison
    rng = np.random.RandomState(seed)
    pq_idx = rng.choice(len(V), min(2000, len(V)), replace=False)
    pq, pq_gt = V[pq_idx], assignments[pq_idx]

    rows_out = []
    for name in cfg['methods']:
        idx = build_index(name, seed)
        t0 = time.time()
        idx.fit(V, assignments)
        build_t = time.time() - t0
        reps, r2s = rep_arrays(idx)

        routed = route_topk_segments(queries, reps, r2s, max(K_EVAL))
        routed_pq = route_topk_segments(pq, reps, r2s, max(K_EVAL))
        row = {'strategy': strategy, 'seed': seed, 'method': name,
               'n_reps': len(reps), 'build_time_s': build_t,
               'n_queries_real': len(queries)}
        for k in K_EVAL:
            row[f'real_recall@{k}'] = routing_recall_multi_gt(
                routed, gt_seg_sets, k)
            row[f'pseudo_recall@{k}'] = routing_recall(routed_pq, pq_gt, k)
        # end-to-end
        e2e_n = min(E2E_MAX_QUERIES, len(queries))
        e2e_sel = rng.choice(len(queries), e2e_n, replace=False)
        for npb in NPROBE_E2E:
            row[f'e2e_hit10@nprobe{npb}'] = e2e_hit_at(
                V, assignments, seg_rows, routed[e2e_sel][:, :npb],
                queries[e2e_sel], [gt_rows[i] for i in e2e_sel])
        rows_out.append(row)
        print(f"[{strategy} seed={seed}] {name}: "
              f"real R@10={row['real_recall@10']:.3f} "
              f"pseudo R@10={row['pseudo_recall@10']:.3f} "
              f"e2e@np5={row['e2e_hit10@nprobe5']:.3f}", flush=True)
    return rows_out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bundle', required=True)
    ap.add_argument('--dataset-name', required=True)
    ap.add_argument('--output', default='revision_results/exp14')
    ap.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44])
    ap.add_argument('--strategies', nargs='+',
                    default=['semi', 'timebatch'])
    ap.add_argument('--methods', nargs='+',
                    default=['Mean', 'HMRC-3', 'HMRC-4', 'kRt-b2d2'])
    ap.add_argument('--n-topics', type=int, default=100)
    ap.add_argument('--segment-size', type=int, default=500)
    args = ap.parse_args()

    bundle = load_bundle(args.bundle)
    cfg = {'n_topics': args.n_topics, 'segment_size': args.segment_size,
           'methods': args.methods}

    rows = []
    for st in args.strategies:
        for sd in args.seeds:
            rows.extend(run_one(bundle, st, sd, cfg))

    df = pd.DataFrame(rows)
    out = f"{args.output}_{args.dataset_name}"
    os.makedirs(out, exist_ok=True)
    raw_path = os.path.join(out, 'exp14_raw.csv')
    if os.path.exists(raw_path):
        prev = pd.read_csv(raw_path)
        df = pd.concat([prev, df], ignore_index=True).drop_duplicates(
            subset=['strategy', 'seed', 'method'], keep='last')
    df.to_csv(raw_path, index=False)
    value_cols = [c for c in df.columns
                  if c.startswith(('real_', 'pseudo_', 'e2e_'))]
    agg = agg_mean_std(df, ['strategy', 'method'], value_cols)
    agg.to_csv(os.path.join(out, 'exp14_aggregated.csv'), index=False)
    print(agg.to_string(index=False))


if __name__ == '__main__':
    main()
