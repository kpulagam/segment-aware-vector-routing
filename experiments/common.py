"""
Shared utilities for HMRC revision experiments (exp12+).

Provides memory-efficient routing, the standard segmentation protocol used
by run_paper_experiments.py, and multi-seed aggregation helpers.
"""

import os
import sys
import time
import warnings
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

warnings.filterwarnings('ignore')


# ----------------------------------------------------------------------------
# Efficient routing (matmul-based distances, chunked over queries)
# ----------------------------------------------------------------------------

def pairwise_sq_dists(queries: np.ndarray, reps: np.ndarray,
                      chunk: int = 256) -> np.ndarray:
    """Squared L2 distances, computed in chunks. (n_q, n_reps) float32."""
    queries = np.ascontiguousarray(queries, dtype=np.float32)
    reps = np.ascontiguousarray(reps, dtype=np.float32)
    rep_sq = (reps ** 2).sum(axis=1)
    out = np.empty((len(queries), len(reps)), dtype=np.float32)
    for s in range(0, len(queries), chunk):
        q = queries[s:s + chunk]
        out[s:s + chunk] = ((q ** 2).sum(axis=1)[:, None]
                            + rep_sq[None, :] - 2.0 * (q @ reps.T))
    return out


def route_topk_segments(queries: np.ndarray, reps: np.ndarray,
                        rep_to_segment: np.ndarray, k: int) -> np.ndarray:
    """Route each query to top-k distinct segments by nearest representative."""
    dists = pairwise_sq_dists(queries, reps)
    order = np.argsort(dists, axis=1)
    n_q = len(queries)
    result = np.full((n_q, k), -1, dtype=np.int64)
    for i in range(n_q):
        seen = set()
        j = 0
        for idx in order[i]:
            seg = rep_to_segment[idx]
            if seg not in seen:
                result[i, j] = seg
                seen.add(seg)
                j += 1
                if j >= k:
                    break
    return result


def routing_recall(routed: np.ndarray, gt_segments: np.ndarray,
                   k: int) -> float:
    """Fraction of queries whose ground-truth segment is in the top-k."""
    hits = sum(1 for i, gt in enumerate(gt_segments) if gt in routed[i, :k])
    return hits / len(gt_segments)


def routing_recall_multi_gt(routed: np.ndarray, gt_sets: List[set],
                            k: int) -> float:
    """Recall when a query has a *set* of acceptable ground-truth segments
    (real-query evaluation: any segment holding a relevant doc counts)."""
    hits = sum(1 for i, gts in enumerate(gt_sets)
               if gts and gts.intersection(routed[i, :k]))
    denom = sum(1 for gts in gt_sets if gts)
    return hits / max(denom, 1)


# ----------------------------------------------------------------------------
# Standard experimental protocol (mirrors run_paper_experiments.py)
# ----------------------------------------------------------------------------

def make_topic_labels(vectors: np.ndarray, n_topics: int = 100,
                      seed: int = 42) -> np.ndarray:
    km = MiniBatchKMeans(n_clusters=n_topics, random_state=seed,
                         batch_size=1024, n_init=3)
    return km.fit_predict(vectors)


def make_segments(vectors: np.ndarray, topic_labels: np.ndarray,
                  strategy: str, segment_size: int = 500,
                  topics_per_segment: int = 3, seed: int = 42) -> np.ndarray:
    """Build segment assignments with the named strategy."""
    from src.segmentation import (create_coherent_segments,
                                  create_random_segments,
                                  create_semi_structured_segments)
    n = len(vectors)
    if strategy == 'coherent':
        return create_coherent_segments(vectors, n // segment_size, seed=seed)
    if strategy == 'random':
        return create_random_segments(n, segment_size, seed=seed)
    if strategy == 'semi':
        return create_semi_structured_segments(
            vectors, topic_labels, segment_size,
            topics_per_segment=topics_per_segment, seed=seed)
    if strategy == 'timebatch':
        # Ingestion-order partitioning (dataset order = arrival order).
        n_segments = n // segment_size
        assignments = np.full(n, -1, dtype=np.int64)
        for seg_id in range(n_segments):
            assignments[seg_id * segment_size:(seg_id + 1) * segment_size] = seg_id
        assignments[n_segments * segment_size:] = max(n_segments - 1, 0)
        return assignments
    raise ValueError(strategy)


def sample_queries(vectors: np.ndarray, assignments: np.ndarray,
                   n_queries: int = 2000,
                   seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """In-corpus query protocol used by the original paper experiments."""
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(vectors), n_queries, replace=False)
    return vectors[idx], assignments[idx]


# ----------------------------------------------------------------------------
# Multi-seed aggregation
# ----------------------------------------------------------------------------

def run_seeds(fn: Callable[[int], Dict], seeds: List[int]) -> pd.DataFrame:
    """Run fn(seed) -> dict of scalars; return long-form DataFrame."""
    rows = []
    for s in seeds:
        r = fn(s)
        r['seed'] = s
        rows.append(r)
    return pd.DataFrame(rows)


def agg_mean_std(df: pd.DataFrame, group_cols: List[str],
                 value_cols: List[str]) -> pd.DataFrame:
    """Aggregate seed runs to mean/std/95% CI half-width."""
    g = df.groupby(group_cols)[value_cols]
    mean = g.mean().add_suffix('_mean')
    std = g.std(ddof=1).add_suffix('_std')
    n = g.count().iloc[:, 0]
    out = mean.join(std)
    for c in value_cols:
        out[c + '_ci95'] = 1.96 * out[c + '_std'] / np.sqrt(n)
    return out.reset_index()
