#!/usr/bin/env python3
"""
Experiment 11: FAISS IVF Comparison (nprobe sweep)
===================================================

Validates HMRC's routing-quality story inside a real production-grade IVF index
(faiss.IndexIVFFlat), addressing the reviewer concern that the existing
extended-k / nprobe comparison (exp7) is run against a NumPy reference rather
than a real ANN index.

For both Baseline (mean centroids) and HMRC-{2,3,5}, we:
  1. Build a faiss.IndexIVFFlat whose coarse quantizer is seeded with our
     externally-defined centroids.
  2. Force each vector into the inverted list corresponding to its segment
     (or, for HMRC, the nearest representative within its segment), bypassing
     FAISS's own assignment.
  3. Sweep nprobe in {1, 3, 5, 10, 15, 20, 30} and measure:
       - Routing Recall@nprobe (does the query's true segment get probed?)
       - End-to-end Recall@10 (does the true nearest neighbor come back?)
       - Search latency (microseconds per query, single-threaded)

Run separately from the main experiment suite — no need to re-run everything.

Usage:
    python exp11_faiss_nprobe.py --embeddings data/embeddings/msmarco_100000.npy \\
        --output paper_results --dataset-name MSMARCO-100K
    python exp11_faiss_nprobe.py --embeddings data/embeddings/nq_dataset.npy \\
        --output paper_results --dataset-name NQ
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '1'

import warnings
warnings.filterwarnings('ignore')

import argparse
import json
import time
from datetime import datetime
import numpy as np
np.seterr(all='ignore')

import pandas as pd
import faiss
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.size'] = 11

from sklearn.cluster import MiniBatchKMeans, KMeans

# Single-threaded everywhere (matches exp10's timing convention).
faiss.omp_set_num_threads(1)


# ============================================================================
# Index Implementations (same as run_paper_experiments.py — kept inline so
# this script is self-contained, matching the project convention)
# ============================================================================

class MeanCentroidIndex:
    """Baseline: single mean centroid per segment."""

    def __init__(self):
        self.centroids = None
        self.segment_ids = None

    def fit(self, vectors, assignments):
        unique_segments = np.unique(assignments)
        centroids, segment_ids = [], []
        for seg_id in unique_segments:
            seg_vectors = vectors[assignments == seg_id]
            if len(seg_vectors) > 0:
                centroids.append(seg_vectors.mean(axis=0))
                segment_ids.append(seg_id)
        self.centroids = np.array(centroids, dtype=np.float32)
        self.segment_ids = np.array(segment_ids)
        return self


class HMRCIndex:
    """HMRC: r centroids per segment via intra-segment k-means."""

    def __init__(self, n_representatives=3):
        self.n_representatives = n_representatives
        self.all_representatives = None
        self.rep_to_segment = None
        self.segment_ids = None

    def fit(self, vectors, assignments):
        unique_segments = np.unique(assignments)
        all_reps, rep_to_seg = [], []
        for seg_id in unique_segments:
            seg_vectors = vectors[assignments == seg_id].astype(np.float64)
            n_vectors = len(seg_vectors)
            if n_vectors < self.n_representatives * 2:
                reps = [seg_vectors.mean(axis=0).astype(np.float32)]
            else:
                k = min(self.n_representatives, n_vectors // 2)
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        km = KMeans(n_clusters=k, random_state=42,
                                    n_init=3, max_iter=100)
                        km.fit(seg_vectors)
                        reps = [c.astype(np.float32) for c in km.cluster_centers_]
                except Exception:
                    reps = [seg_vectors.mean(axis=0).astype(np.float32)]
            all_reps.extend(reps)
            rep_to_seg.extend([seg_id] * len(reps))
        self.all_representatives = np.array(all_reps, dtype=np.float32)
        self.rep_to_segment = np.array(rep_to_seg)
        self.segment_ids = unique_segments
        return self


# ============================================================================
# Segment creation (copied from run_paper_experiments.py)
# ============================================================================

def create_topic_clusters(vectors, n_topics, seed=42):
    km = MiniBatchKMeans(n_clusters=n_topics, random_state=seed,
                         batch_size=1024, n_init=3)
    return km.fit_predict(vectors)


def create_semi_structured_segments(vectors, topic_labels, segment_size,
                                    topics_per_segment=3, seed=42):
    np.random.seed(seed)
    n_vectors = len(vectors)
    n_topics = len(np.unique(topic_labels))
    n_segments = n_vectors // segment_size

    topic_to_vectors = {}
    for i, t in enumerate(topic_labels):
        topic_to_vectors.setdefault(t, []).append(i)
    for t in topic_to_vectors:
        np.random.shuffle(topic_to_vectors[t])

    topic_pos = {t: 0 for t in topic_to_vectors}
    assignments = np.full(n_vectors, -1, dtype=np.int32)

    for seg_id in range(n_segments):
        selected = np.random.choice(list(topic_to_vectors.keys()),
                                    size=min(topics_per_segment, n_topics),
                                    replace=False)
        per_topic = segment_size // len(selected)
        rem = segment_size % len(selected)
        for i, t in enumerate(selected):
            n_take = per_topic + (1 if i < rem else 0)
            tv = topic_to_vectors[t]
            start = topic_pos[t]
            end = min(start + n_take, len(tv))
            for idx in tv[start:end]:
                assignments[idx] = seg_id
            topic_pos[t] = end if end < len(tv) else 0

    unassigned = np.where(assignments == -1)[0]
    for idx in unassigned:
        assignments[idx] = np.random.randint(n_segments)
    return assignments


# ============================================================================
# FAISS index construction with externally-defined inverted lists
# ============================================================================

def build_faiss_ivf(vectors, list_centroids, vector_to_list, d):
    """
    Build a faiss.IndexIVFFlat where:
      - The coarse quantizer is pre-loaded with `list_centroids`.
      - Each vector goes into the inverted list given by `vector_to_list[i]`.
    """
    n_lists = len(list_centroids)

    quantizer = faiss.IndexFlatL2(d)
    quantizer.add(list_centroids.astype(np.float32))

    index = faiss.IndexIVFFlat(quantizer, d, n_lists, faiss.METRIC_L2)
    index.is_trained = True
    index.nprobe = 1

    inv_lists = index.invlists
    vectors_f32 = np.ascontiguousarray(vectors.astype(np.float32))
    n_total = len(vectors_f32)
    ids_all = np.arange(n_total, dtype=np.int64)

    for list_id in range(n_lists):
        mask = vector_to_list == list_id
        if not mask.any():
            continue
        sub_vecs = np.ascontiguousarray(vectors_f32[mask])
        sub_ids = np.ascontiguousarray(ids_all[mask])
        # InvertedLists.add_entries takes uint8_t* (generic codec interface).
        # IVFFlat codes ARE float32, so reinterpret the buffer as bytes.
        sub_vecs_bytes = sub_vecs.view(np.uint8).reshape(-1)
        inv_lists.add_entries(
            list_id,
            len(sub_ids),
            faiss.swig_ptr(sub_ids),
            faiss.swig_ptr(sub_vecs_bytes),
        )

    # Sync ntotal from what was actually added
    total_added = sum(inv_lists.list_size(i) for i in range(n_lists))
    index.ntotal = total_added
    return index


def assign_to_nearest_representative(vectors, assignments, hmrc_index):
    """
    For each vector, return the *global* representative ID (i.e., the index
    into hmrc_index.all_representatives) of the closest representative
    *within its own segment*. This is the FAISS inverted-list ID for the
    HMRC-augmented index.
    """
    n = len(vectors)
    rep_assignment = np.zeros(n, dtype=np.int64)

    # Group representatives by segment for fast within-segment lookup
    seg_to_rep_ids = {}
    for global_rep_id, seg_id in enumerate(hmrc_index.rep_to_segment):
        seg_to_rep_ids.setdefault(seg_id, []).append(global_rep_id)

    for seg_id, rep_ids in seg_to_rep_ids.items():
        rep_ids = np.array(rep_ids)
        seg_reps = hmrc_index.all_representatives[rep_ids]
        seg_vec_mask = assignments == seg_id
        seg_vecs = vectors[seg_vec_mask].astype(np.float32)
        if len(seg_vecs) == 0:
            continue
        # nearest representative within segment
        dists = np.linalg.norm(seg_vecs[:, None, :] - seg_reps[None, :, :], axis=2)
        nearest = np.argmin(dists, axis=1)
        rep_assignment[seg_vec_mask] = rep_ids[nearest]
    return rep_assignment


# ============================================================================
# Evaluation
# ============================================================================

def true_nearest_neighbor(vectors, queries):
    """Brute-force NN ground truth via FAISS IndexFlatL2 (memory-safe)."""
    d = vectors.shape[1]
    flat = faiss.IndexFlatL2(d)
    flat.add(np.ascontiguousarray(vectors.astype(np.float32)))
    _, I = flat.search(np.ascontiguousarray(queries.astype(np.float32)), 1)
    return I[:, 0].astype(np.int64)


def evaluate_index(index, queries, gt_segments, gt_neighbors,
                   list_to_segment, nprobe_values, k_search=10):
    """
    For each nprobe value, run FAISS search and compute three metrics:
      - routing_recall: did the query's true segment get probed?
      - e2e_recall: is the true nearest neighbor in the top-k results?
      - latency_us: average per-query search time in microseconds
    """
    rows = []
    queries_f32 = np.ascontiguousarray(queries.astype(np.float32))
    n_q = len(queries_f32)

    for nprobe in nprobe_values:
        index.nprobe = nprobe

        # Latency (warmup + 5 timed runs)
        for _ in range(2):
            _ = index.search(queries_f32, k_search)
        times = []
        for _ in range(5):
            t0 = time.perf_counter()
            D, I = index.search(queries_f32, k_search)
            times.append(time.perf_counter() - t0)
        latency_us = (np.mean(times) / n_q) * 1e6

        # End-to-end Recall@k_search: ground-truth NN appears in returned IDs
        e2e_hits = sum(1 for i in range(n_q) if gt_neighbors[i] in I[i])
        e2e_recall = e2e_hits / n_q

        # Routing recall: which inverted lists were actually probed?
        # We re-run quantizer.search(queries, nprobe) to get the probed lists,
        # then map list_id -> segment_id and check membership.
        _, probed_lists = index.quantizer.search(queries_f32, nprobe)
        # probed_lists shape: (n_q, nprobe), values are list IDs
        probed_segments = list_to_segment[probed_lists]  # (n_q, nprobe)
        routing_hits = 0
        for i in range(n_q):
            if gt_segments[i] in probed_segments[i]:
                routing_hits += 1
        routing_recall = routing_hits / n_q

        rows.append({
            'nprobe': nprobe,
            'routing_recall': routing_recall,
            'e2e_recall': e2e_recall,
            'latency_us_per_query': latency_us,
        })

    return rows


# ============================================================================
# Main experiment
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Experiment 11: FAISS IVF nprobe comparison')
    parser.add_argument('--embeddings', type=str, required=True)
    parser.add_argument('--output', type=str, default='paper_results')
    parser.add_argument('--dataset-name', type=str, default='MSMARCO-100K')
    parser.add_argument('--segment-size', type=int, default=500)
    parser.add_argument('--n-topics', type=int, default=100)
    parser.add_argument('--n-queries', type=int, default=2000)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"{args.output}_exp11_{args.dataset_name}_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 70)
    print("EXPERIMENT 11: FAISS IVF nprobe Comparison")
    print("=" * 70)
    print(f"Dataset: {args.dataset_name}")
    print(f"Output:  {output_dir}")
    print(f"FAISS version: {faiss.__version__}")

    # Load + normalize
    print(f"\n  Loading embeddings from {args.embeddings}...")
    vectors = np.load(args.embeddings).astype(np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors = vectors / (norms + 1e-10)
    n, d = vectors.shape
    print(f"   Shape: {vectors.shape}")

    # Topic clusters & Semi-3 segments
    print(f"  Creating {args.n_topics} topic clusters...")
    topic_labels = create_topic_clusters(vectors, args.n_topics, args.seed)
    assignments = create_semi_structured_segments(
        vectors, topic_labels, args.segment_size,
        topics_per_segment=3, seed=args.seed)
    n_segments = len(np.unique(assignments))
    print(f"  Segments: {n_segments} (Semi-3, size {args.segment_size})")

    # Queries + ground truth
    np.random.seed(args.seed)
    query_idx = np.random.choice(n, args.n_queries, replace=False)
    queries = vectors[query_idx].astype(np.float32)
    gt_segments = assignments[query_idx]

    # The query's own vector is its own NN (distance 0). That's not useful
    # for measuring e2e recall. Instead, hold out queries from the corpus.
    corpus_mask = np.ones(n, dtype=bool)
    corpus_mask[query_idx] = False
    corpus_vectors = vectors[corpus_mask]
    corpus_assignments = assignments[corpus_mask]
    # Map old corpus indices -> new corpus indices for ground-truth IDs
    print(f"  Corpus: {len(corpus_vectors)} vectors (queries held out)")
    print(f"  Computing brute-force NN ground truth...")
    gt_neighbors_corpus_idx = true_nearest_neighbor(corpus_vectors, queries)
    print(f"  Done.\n")

    nprobe_values = [1, 3, 5, 10, 15, 20, 30]
    all_rows = []

    # ----- Baseline: 1 list per segment -----
    print("--- Baseline (mean centroid per segment) ---")
    baseline = MeanCentroidIndex().fit(corpus_vectors, corpus_assignments)
    # list_id == segment_id (segments are 0..n_segments-1)
    list_to_segment_base = baseline.segment_ids.astype(np.int64)
    vec_to_list_base = corpus_assignments.astype(np.int64)
    idx_base = build_faiss_ivf(corpus_vectors, baseline.centroids,
                               vec_to_list_base, d)
    rows = evaluate_index(idx_base, queries, gt_segments,
                          gt_neighbors_corpus_idx,
                          list_to_segment_base, nprobe_values)
    for r in rows:
        r['method'] = 'Baseline'
        r['n_lists'] = n_segments
        all_rows.append(r)
        print(f"  nprobe={r['nprobe']:>2}  routing={r['routing_recall']:.4f}"
              f"  e2e@10={r['e2e_recall']:.4f}"
              f"  {r['latency_us_per_query']:.1f} us/q")

    # ----- HMRC variants -----
    for r_value in [2, 3, 5]:
        print(f"\n--- HMRC-{r_value} ---")
        hmrc = HMRCIndex(n_representatives=r_value).fit(
            corpus_vectors, corpus_assignments)
        list_to_segment_hmrc = hmrc.rep_to_segment.astype(np.int64)
        vec_to_list_hmrc = assign_to_nearest_representative(
            corpus_vectors, corpus_assignments, hmrc)
        idx_hmrc = build_faiss_ivf(corpus_vectors,
                                   hmrc.all_representatives,
                                   vec_to_list_hmrc, d)
        rows = evaluate_index(idx_hmrc, queries, gt_segments,
                              gt_neighbors_corpus_idx,
                              list_to_segment_hmrc, nprobe_values)
        for r in rows:
            r['method'] = f'HMRC-{r_value}'
            r['n_lists'] = len(hmrc.all_representatives)
            all_rows.append(r)
            print(f"  nprobe={r['nprobe']:>2}  routing={r['routing_recall']:.4f}"
                  f"  e2e@10={r['e2e_recall']:.4f}"
                  f"  {r['latency_us_per_query']:.1f} us/q")

    # ============ Save CSV ============
    df = pd.DataFrame(all_rows)
    csv_path = os.path.join(output_dir, 'exp11_faiss_nprobe.csv')
    df.to_csv(csv_path, index=False)
    print(f"\n  CSV saved to {csv_path}")

    # ============ Figure 9: routing recall vs nprobe ============
    fig, ax = plt.subplots(figsize=(9, 6))
    styles = {
        'Baseline': ('--o', '#e74c3c'),
        'HMRC-2':   ('-^', '#95a5a6'),
        'HMRC-3':   ('-s', '#3498db'),
        'HMRC-5':   ('-D', '#2ecc71'),
    }
    for method, (style, color) in styles.items():
        sub = df[df['method'] == method].sort_values('nprobe')
        if len(sub) == 0:
            continue
        ax.plot(sub['nprobe'], sub['routing_recall'], style,
                label=method, linewidth=2.5, markersize=8, color=color)
    ax.set_xlabel('nprobe (lists searched)', fontsize=12)
    ax.set_ylabel('Routing Recall', fontsize=12)
    ax.set_title(f'FAISS IVF: Routing Recall vs nprobe ({args.dataset_name}, Semi-3)',
                 fontsize=13)
    ax.legend(fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.set_xticks(nprobe_values)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    for ext in ('pdf', 'png'):
        plt.savefig(os.path.join(output_dir, f'fig9_faiss_routing.{ext}'),
                    dpi=300, bbox_inches='tight')
    plt.close()

    # ============ Figure 10: end-to-end Recall@10 vs nprobe ============
    fig, ax = plt.subplots(figsize=(9, 6))
    for method, (style, color) in styles.items():
        sub = df[df['method'] == method].sort_values('nprobe')
        if len(sub) == 0:
            continue
        ax.plot(sub['nprobe'], sub['e2e_recall'], style,
                label=method, linewidth=2.5, markersize=8, color=color)
    ax.set_xlabel('nprobe (lists searched)', fontsize=12)
    ax.set_ylabel('End-to-end Recall@10', fontsize=12)
    ax.set_title(f'FAISS IVF: Retrieval Recall@10 vs nprobe ({args.dataset_name}, Semi-3)',
                 fontsize=13)
    ax.legend(fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.set_xticks(nprobe_values)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    for ext in ('pdf', 'png'):
        plt.savefig(os.path.join(output_dir, f'fig10_faiss_e2e.{ext}'),
                    dpi=300, bbox_inches='tight')
    plt.close()

    # ============ Summary for paper ============
    print("\n" + "=" * 70)
    print(f"SUMMARY FOR PAPER — {args.dataset_name}")
    print("=" * 70)
    pivot_routing = df.pivot(index='method', columns='nprobe',
                             values='routing_recall').round(4)
    pivot_e2e = df.pivot(index='method', columns='nprobe',
                         values='e2e_recall').round(4)
    pivot_lat = df.pivot(index='method', columns='nprobe',
                         values='latency_us_per_query').round(1)
    print("\n  Routing Recall by nprobe:")
    print(pivot_routing.to_string())
    print("\n  End-to-end Recall@10 by nprobe:")
    print(pivot_e2e.to_string())
    print("\n  Latency (us per query) by nprobe:")
    print(pivot_lat.to_string())

    # Key comparison for the paper write-up
    try:
        b30 = df[(df['method'] == 'Baseline') & (df['nprobe'] == 30)]['e2e_recall'].values[0]
        h10 = df[(df['method'] == 'HMRC-3') & (df['nprobe'] == 10)]['e2e_recall'].values[0]
        print(f"\n  Key result: HMRC-3 @ nprobe=10 → E2E Recall@10 = {h10:.4f}")
        print(f"              Baseline @ nprobe=30 → E2E Recall@10 = {b30:.4f}")
    except (IndexError, KeyError):
        pass

    summary = {
        'dataset': args.dataset_name,
        'faiss_version': faiss.__version__,
        'n_corpus_vectors': int(len(corpus_vectors)),
        'n_queries': int(len(queries)),
        'embedding_dim': int(d),
        'n_segments': int(n_segments),
        'segment_size': int(args.segment_size),
        'nprobe_values': nprobe_values,
    }
    with open(os.path.join(output_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n  All results saved to: {output_dir}/")


if __name__ == "__main__":
    main()
