"""
Competing baselines for HMRC comparison.

KRtStyleIndex: A faithful adaptation of the kRt sub-cluster routing method
(introduced with the Pyramid partitioning system, VLDB 2025) to the
fixed-segment setting. kRt sub-clusters the points within each shard using
hierarchical k-means and routes queries to the shards whose sub-cluster
centers are nearest.

Differences from HMRC:
  - Representatives are produced by *hierarchical* (recursive) k-means with
    branching factor b and depth d (leaves = b^d sub-centers per segment),
    rather than a single flat k-means with small k.
  - The number of sub-centers per segment is typically larger (Pyramid's
    regime: numerous, semantically tight sub-clusters).

Routing here performs an exact nearest-neighbor scan over all sub-centers
(recall-favorable to kRt; Pyramid's HNSW-over-sub-centers is an
approximation of this). Routing cost is reported as the number of
representative comparisons so cost-normalized comparisons are possible.
"""

import warnings
from typing import Dict, Optional

import numpy as np
from sklearn.cluster import KMeans


def _hierarchical_kmeans(vectors: np.ndarray, branching: int, depth: int,
                         seed: int = 42) -> np.ndarray:
    """Recursive k-means. Returns leaf centers, shape (n_leaves, dim)."""
    if depth == 0 or len(vectors) < 2 * branching:
        return vectors.mean(axis=0, keepdims=True)
    k = min(branching, max(1, len(vectors) // 2))
    if k == 1:
        return vectors.mean(axis=0, keepdims=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        km = KMeans(n_clusters=k, random_state=seed, n_init=3, max_iter=100)
        labels = km.fit_predict(vectors)
    if depth == 1:
        return km.cluster_centers_
    leaves = []
    for c in range(k):
        sub = vectors[labels == c]
        if len(sub) == 0:
            continue
        leaves.append(_hierarchical_kmeans(sub, branching, depth - 1, seed))
    return np.vstack(leaves)


class KRtStyleIndex:
    """kRt-style sub-cluster routing baseline (hierarchical k-means)."""

    def __init__(self, branching: int = 4, depth: int = 2,
                 min_segment_size: int = 10, seed: int = 42):
        self.branching = branching
        self.depth = depth
        self.min_segment_size = min_segment_size
        self.seed = seed
        self.all_representatives: Optional[np.ndarray] = None
        self.rep_to_segment: Optional[np.ndarray] = None
        self.segment_ids: Optional[np.ndarray] = None
        self.n_segments: int = 0

    def fit(self, vectors: np.ndarray, assignments: np.ndarray) -> 'KRtStyleIndex':
        unique_segments = np.unique(assignments)
        all_reps, rep_to_seg = [], []
        for seg_id in unique_segments:
            seg_vectors = vectors[assignments == seg_id].astype(np.float64)
            if len(seg_vectors) < self.min_segment_size:
                reps = seg_vectors.mean(axis=0, keepdims=True)
            else:
                reps = _hierarchical_kmeans(seg_vectors, self.branching,
                                            self.depth, self.seed)
            reps = reps.astype(np.float32)
            all_reps.append(reps)
            rep_to_seg.extend([seg_id] * len(reps))
        self.all_representatives = np.vstack(all_reps)
        self.rep_to_segment = np.array(rep_to_seg)
        self.segment_ids = unique_segments
        self.n_segments = len(unique_segments)
        return self

    def route(self, queries: np.ndarray, k: int = 10) -> np.ndarray:
        """Route queries to top-k segments (exact scan over sub-centers,
        deduplicated by segment)."""
        queries = np.asarray(queries, dtype=np.float32)
        n_queries = len(queries)
        dists = np.linalg.norm(
            queries[:, None, :] - self.all_representatives[None, :, :], axis=2)
        order = np.argsort(dists, axis=1)
        result = np.full((n_queries, k), -1, dtype=np.int64)
        for i in range(n_queries):
            seen = set()
            j = 0
            for idx in order[i]:
                seg_id = self.rep_to_segment[idx]
                if seg_id not in seen:
                    result[i, j] = seg_id
                    seen.add(seg_id)
                    j += 1
                    if j >= k:
                        break
        return result

    def get_stats(self) -> Dict:
        avg = len(self.all_representatives) / max(self.n_segments, 1)
        return {
            'n_segments': self.n_segments,
            'n_representatives': len(self.all_representatives),
            'avg_reps_per_segment': avg,
            'memory_overhead': avg,
            'method': f'kRt-b{self.branching}d{self.depth}',
        }
