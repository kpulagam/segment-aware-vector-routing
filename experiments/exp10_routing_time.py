#!/usr/bin/env python3
"""
Experiment 10: Query Routing Time Measurement
==============================================

Measures per-query routing latency for Baseline vs HMRC variants.
Run separately from the main experiment suite — no need to re-run everything.

Usage:
    python exp10_routing_time.py --embeddings /path/to/msmarco_100000.npy
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '1'

import warnings
warnings.filterwarnings('ignore')

import argparse
import time
import numpy as np
np.seterr(all='ignore')

from sklearn.cluster import MiniBatchKMeans, KMeans


# ============================================================================
# Index Implementations (same as main script)
# ============================================================================

class MeanCentroidIndex:
    def __init__(self):
        self.centroids = None
        self.segment_ids = None

    def fit(self, vectors, assignments):
        unique_segments = np.unique(assignments)
        centroids = []
        segment_ids = []
        for seg_id in unique_segments:
            mask = assignments == seg_id
            seg_vectors = vectors[mask]
            if len(seg_vectors) > 0:
                centroids.append(seg_vectors.mean(axis=0))
                segment_ids.append(seg_id)
        self.centroids = np.array(centroids, dtype=np.float32)
        self.segment_ids = np.array(segment_ids)
        return self

    def route(self, queries, k=10):
        queries = np.asarray(queries, dtype=np.float32)
        dists = np.linalg.norm(queries[:, None, :] - self.centroids[None, :, :], axis=2)
        top_k_indices = np.argsort(dists, axis=1)[:, :k]
        return self.segment_ids[top_k_indices]

    def get_n_centroids(self):
        return len(self.centroids)


class HMRCIndex:
    def __init__(self, n_representatives=3):
        self.n_representatives = n_representatives
        self.all_representatives = None
        self.rep_to_segment = None
        self.segment_ids = None

    def fit(self, vectors, assignments):
        unique_segments = np.unique(assignments)
        all_reps = []
        rep_to_seg = []
        for seg_id in unique_segments:
            mask = assignments == seg_id
            seg_vectors = vectors[mask].astype(np.float64)
            n_vectors = len(seg_vectors)
            if n_vectors < self.n_representatives * 2:
                reps = [seg_vectors.mean(axis=0).astype(np.float32)]
            else:
                k = min(self.n_representatives, n_vectors // 2)
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        kmeans = KMeans(n_clusters=k, random_state=42, n_init=3, max_iter=100)
                        kmeans.fit(seg_vectors)
                        reps = [c.astype(np.float32) for c in kmeans.cluster_centers_]
                except:
                    reps = [seg_vectors.mean(axis=0).astype(np.float32)]
            all_reps.extend(reps)
            rep_to_seg.extend([seg_id] * len(reps))
        self.all_representatives = np.array(all_reps, dtype=np.float32)
        self.rep_to_segment = np.array(rep_to_seg)
        self.segment_ids = unique_segments
        return self

    def route(self, queries, k=10):
        queries = np.asarray(queries, dtype=np.float32)
        n_queries = len(queries)
        dists = np.linalg.norm(
            queries[:, None, :] - self.all_representatives[None, :, :], axis=2
        )
        n_search = min(k * self.n_representatives * 2, len(self.all_representatives))
        top_indices = np.argsort(dists, axis=1)[:, :n_search]
        result = np.zeros((n_queries, k), dtype=np.int64)
        for i in range(n_queries):
            seen = set()
            j = 0
            for idx in top_indices[i]:
                seg_id = self.rep_to_segment[idx]
                if seg_id not in seen:
                    result[i, j] = seg_id
                    seen.add(seg_id)
                    j += 1
                    if j >= k:
                        break
        return result

    def get_n_centroids(self):
        return len(self.all_representatives)


# ============================================================================
# Segment Creation
# ============================================================================

def create_semi_structured_segments(vectors, topic_labels, segment_size, topics_per_segment=3, seed=42):
    np.random.seed(seed)
    n_vectors = len(vectors)
    n_topics = len(np.unique(topic_labels))
    n_segments = n_vectors // segment_size

    topic_to_vectors = {}
    for i, topic in enumerate(topic_labels):
        if topic not in topic_to_vectors:
            topic_to_vectors[topic] = []
        topic_to_vectors[topic].append(i)
    for topic in topic_to_vectors:
        np.random.shuffle(topic_to_vectors[topic])

    topic_positions = {t: 0 for t in topic_to_vectors}
    assignments = np.full(n_vectors, -1, dtype=np.int32)

    for seg_id in range(n_segments):
        selected_topics = np.random.choice(
            list(topic_to_vectors.keys()),
            size=min(topics_per_segment, n_topics),
            replace=False
        )
        vectors_per_topic = segment_size // len(selected_topics)
        remainder = segment_size % len(selected_topics)
        for i, topic in enumerate(selected_topics):
            n_from_topic = vectors_per_topic + (1 if i < remainder else 0)
            tv = topic_to_vectors[topic]
            start = topic_positions[topic]
            end = min(start + n_from_topic, len(tv))
            for idx in tv[start:end]:
                assignments[idx] = seg_id
            topic_positions[topic] = end
            if topic_positions[topic] >= len(tv):
                topic_positions[topic] = 0

    unassigned = np.where(assignments == -1)[0]
    for idx in unassigned:
        assignments[idx] = np.random.randint(n_segments)
    return assignments


# ============================================================================
# Timing Experiment
# ============================================================================

def measure_routing_time(index, queries, k=10, n_warmup=3, n_runs=10):
    """
    Measure per-query routing time in microseconds.
    Runs warmup iterations first, then averages over n_runs.
    Returns (mean_us_per_query, std_us_per_query, total_ms).
    """
    n_queries = len(queries)

    # Warmup — let CPU caches settle
    for _ in range(n_warmup):
        _ = index.route(queries, k=k)

    # Timed runs
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        _ = index.route(queries, k=k)
        elapsed = time.perf_counter() - t0
        times.append(elapsed)

    times = np.array(times)
    mean_total = np.mean(times)
    std_total = np.std(times)
    mean_per_query_us = (mean_total / n_queries) * 1e6
    std_per_query_us = (std_total / n_queries) * 1e6

    return mean_per_query_us, std_per_query_us, mean_total * 1000


def main():
    parser = argparse.ArgumentParser(description='Experiment 10: Query Routing Time')
    parser.add_argument('--embeddings', type=str, required=True)
    parser.add_argument('--segment-size', type=int, default=500)
    parser.add_argument('--n-topics', type=int, default=100)
    parser.add_argument('--n-queries', type=int, default=2000)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    print("="*70)
    print("EXPERIMENT 10: Query Routing Time Measurement")
    print("="*70)

    # Load and normalize
    print(f"\n  Loading embeddings from {args.embeddings}...")
    vectors = np.load(args.embeddings).astype(np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors = vectors / (norms + 1e-10)
    print(f"  Shape: {vectors.shape}")

    # Topic clusters
    print(f"  Creating {args.n_topics} topic clusters...")
    kmeans = MiniBatchKMeans(n_clusters=args.n_topics, random_state=args.seed, batch_size=1024, n_init=3)
    topic_labels = kmeans.fit_predict(vectors)

    # Semi-3 segments
    assignments = create_semi_structured_segments(
        vectors, topic_labels, args.segment_size,
        topics_per_segment=3, seed=args.seed
    )
    n_segments = len(np.unique(assignments))

    # Queries
    np.random.seed(args.seed)
    query_idx = np.random.choice(len(vectors), args.n_queries, replace=False)
    query_vectors = vectors[query_idx].astype(np.float32)

    print(f"  Segments: {n_segments}, Queries: {args.n_queries}")
    print(f"  Measuring with 3 warmup + 10 timed runs per method\n")

    # Methods
    methods = [
        ('Baseline',  MeanCentroidIndex, {}),
        ('HMRC-2',    HMRCIndex, {'n_representatives': 2}),
        ('HMRC-3',    HMRCIndex, {'n_representatives': 3}),
        ('HMRC-5',    HMRCIndex, {'n_representatives': 5}),
        ('HMRC-7',    HMRCIndex, {'n_representatives': 7}),
    ]

    print(f"  {'Method':<12} {'Centroids':>10} {'Per-Query (μs)':>16} {'± Std (μs)':>12} {'Total (ms)':>12} {'Slowdown':>10}")
    print("  " + "-"*76)

    baseline_us = None

    for method_name, IndexClass, kwargs in methods:
        index = IndexClass(**kwargs)
        index.fit(vectors, assignments)

        mean_us, std_us, total_ms = measure_routing_time(index, query_vectors, k=10)

        if baseline_us is None:
            baseline_us = mean_us
            slowdown_str = "1.0×"
        else:
            slowdown_str = f"{mean_us / baseline_us:.1f}×"

        print(f"  {method_name:<12} {index.get_n_centroids():>10} {mean_us:>14.1f} {std_us:>10.1f} {total_ms:>10.1f} {slowdown_str:>10}")

    # Summary for paper
    print(f"\n" + "="*70)
    print("SUMMARY FOR PAPER")
    print("="*70)
    print(f"""
  On {args.n_queries} queries routed across {n_segments} segments:
  - Baseline routing takes ~X μs per query (Y ms total for {args.n_queries} queries)
  - HMRC-3 routing takes ~X μs per query (Y ms total)
  - The overhead is Z× in routing time — sub-millisecond per query in both cases
  - Within-segment search (not measured here) typically dominates at 10-100ms

  Copy the table above into the paper's Section IV-D (Complexity and Overhead).
""")


if __name__ == "__main__":
    main()
