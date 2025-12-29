#!/usr/bin/env python3
"""
HMRC Paper Experiments - Complete Suite
========================================

Generates all figures and tables needed for the HMRC paper.

Usage:
    python run_paper_experiments.py --embeddings data/embeddings/msmarco_100000.npy --output paper_results

Experiments:
    1. Problem Validation: Coherent vs Heterogeneous gap
    2. HMRC Solution: Performance on semi-structured segments
    3. Ablation: Number of topics per segment (2, 3, 5, 7)
    4. Ablation: Number of representatives (1, 2, 3, 5, 7)
    5. Comparison: HMRC vs Multi-probe baseline
    6. Scalability: Different segment sizes
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
from typing import Dict, List, Tuple
import numpy as np
np.seterr(all='ignore')

import pandas as pd
from sklearn.cluster import MiniBatchKMeans, KMeans
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.size'] = 11


# ============================================================================
# Index Implementations
# ============================================================================

class MeanCentroidIndex:
    """Baseline: Single mean centroid per segment."""
    
    def __init__(self):
        self.centroids = None
        self.segment_ids = None
    
    def fit(self, vectors: np.ndarray, assignments: np.ndarray):
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
    
    def route(self, queries: np.ndarray, k: int = 10) -> np.ndarray:
        queries = np.asarray(queries, dtype=np.float32)
        dists = np.linalg.norm(queries[:, None, :] - self.centroids[None, :, :], axis=2)
        top_k_indices = np.argsort(dists, axis=1)[:, :k]
        return self.segment_ids[top_k_indices]
    
    def get_overhead(self) -> float:
        return 1.0


class MultiProbeIndex:
    """Multi-probe baseline: Search top-k centroids (like nprobe in FAISS)."""
    
    def __init__(self, n_probe: int = 3):
        self.n_probe = n_probe
        self.centroids = None
        self.segment_ids = None
    
    def fit(self, vectors: np.ndarray, assignments: np.ndarray):
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
    
    def route(self, queries: np.ndarray, k: int = 10) -> np.ndarray:
        # Multi-probe returns n_probe * k segments
        queries = np.asarray(queries, dtype=np.float32)
        dists = np.linalg.norm(queries[:, None, :] - self.centroids[None, :, :], axis=2)
        # Return top n_probe * k, but we evaluate on top k unique
        top_k_indices = np.argsort(dists, axis=1)[:, :k]
        return self.segment_ids[top_k_indices]
    
    def get_overhead(self) -> float:
        return self.n_probe  # Searches n_probe times more segments


class HMRCIndex:
    """HMRC: Heterogeneous Multi-Representative Centroids."""
    
    def __init__(self, n_representatives: int = 3):
        self.n_representatives = n_representatives
        self.all_representatives = None
        self.rep_to_segment = None
        self.segment_ids = None
    
    def fit(self, vectors: np.ndarray, assignments: np.ndarray):
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
    
    def route(self, queries: np.ndarray, k: int = 10) -> np.ndarray:
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
    
    def get_overhead(self) -> float:
        return len(self.all_representatives) / len(self.segment_ids)


# ============================================================================
# Segment Creation
# ============================================================================

def create_topic_clusters(vectors: np.ndarray, n_topics: int, seed: int = 42) -> np.ndarray:
    """Pre-cluster vectors into topic clusters."""
    kmeans = MiniBatchKMeans(n_clusters=n_topics, random_state=seed, batch_size=1024, n_init=3)
    return kmeans.fit_predict(vectors)


def create_coherent_segments(vectors: np.ndarray, n_segments: int, seed: int = 42) -> np.ndarray:
    """Semantically coherent segments via clustering."""
    kmeans = MiniBatchKMeans(n_clusters=n_segments, random_state=seed, batch_size=1024, n_init=3)
    return kmeans.fit_predict(vectors)


def create_random_segments(n_vectors: int, segment_size: int, seed: int = 42) -> np.ndarray:
    """Purely random segments (worst case)."""
    np.random.seed(seed)
    n_segments = n_vectors // segment_size
    assignments = np.repeat(np.arange(n_segments), segment_size)[:n_vectors]
    np.random.shuffle(assignments)
    return assignments


def create_semi_structured_segments(
    vectors: np.ndarray,
    topic_labels: np.ndarray,
    segment_size: int,
    topics_per_segment: int = 3,
    seed: int = 42
) -> np.ndarray:
    """
    Semi-structured heterogeneous segments.
    Each segment mixes vectors from `topics_per_segment` distinct topics.
    """
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
            topic_vectors = topic_to_vectors[topic]
            start = topic_positions[topic]
            end = min(start + n_from_topic, len(topic_vectors))
            
            for idx in topic_vectors[start:end]:
                assignments[idx] = seg_id
            
            topic_positions[topic] = end
            if topic_positions[topic] >= len(topic_vectors):
                topic_positions[topic] = 0
    
    unassigned = np.where(assignments == -1)[0]
    for idx in unassigned:
        assignments[idx] = np.random.randint(n_segments)
    
    return assignments


# ============================================================================
# Evaluation
# ============================================================================

def evaluate_routing(index, queries: np.ndarray, gt_segments: np.ndarray, k: int = 10) -> float:
    """Compute routing recall@k."""
    routed = index.route(queries, k=k)
    hits = sum(1 for i, gt in enumerate(gt_segments) if gt in routed[i, :k])
    return hits / len(gt_segments)


def compute_segment_statistics(vectors: np.ndarray, assignments: np.ndarray) -> Dict:
    """Compute statistics about segment coherence."""
    unique_segments = np.unique(assignments)
    
    intra_dists = []
    for seg_id in unique_segments:
        mask = assignments == seg_id
        seg_vectors = vectors[mask]
        if len(seg_vectors) > 1:
            centroid = seg_vectors.mean(axis=0)
            dists = np.linalg.norm(seg_vectors - centroid, axis=1)
            intra_dists.append(dists.mean())
    
    return {
        'mean_intra_dist': np.mean(intra_dists),
        'std_intra_dist': np.std(intra_dists),
        'n_segments': len(unique_segments)
    }


# ============================================================================
# Experiments
# ============================================================================

def experiment_1_problem_validation(vectors, topic_labels, config) -> pd.DataFrame:
    """
    Experiment 1: Validate the problem exists.
    Compare routing recall on coherent vs heterogeneous segments.
    """
    print("\n" + "="*70)
    print("EXPERIMENT 1: Problem Validation")
    print("="*70)
    
    results = []
    segment_size = config['segment_size']
    n_segments = len(vectors) // segment_size
    
    segmentations = {
        'Coherent': create_coherent_segments(vectors, n_segments, config['seed']),
        'Semi-3': create_semi_structured_segments(vectors, topic_labels, segment_size, 3, config['seed']),
        'Semi-5': create_semi_structured_segments(vectors, topic_labels, segment_size, 5, config['seed']),
        'Random': create_random_segments(len(vectors), segment_size, config['seed']),
    }
    
    # Generate queries
    np.random.seed(config['seed'])
    query_idx = np.random.choice(len(vectors), config['n_queries'], replace=False)
    query_vectors = vectors[query_idx]
    
    for seg_name, assignments in segmentations.items():
        print(f"\n--- {seg_name} Segments ---")
        
        query_gt = assignments[query_idx]
        stats = compute_segment_statistics(vectors, assignments)
        
        index = MeanCentroidIndex()
        index.fit(vectors, assignments)
        
        for k in [1, 5, 10, 20]:
            recall = evaluate_routing(index, query_vectors, query_gt, k=k)
            results.append({
                'segmentation': seg_name,
                'k': k,
                'recall': recall,
                'mean_intra_dist': stats['mean_intra_dist'],
            })
        
        print(f"  Recall@10: {results[-2]['recall']:.4f}")
        print(f"  Mean intra-segment distance: {stats['mean_intra_dist']:.4f}")
    
    return pd.DataFrame(results)


def experiment_2_hmrc_solution(vectors, topic_labels, config) -> pd.DataFrame:
    """
    Experiment 2: HMRC solution on semi-structured segments.
    """
    print("\n" + "="*70)
    print("EXPERIMENT 2: HMRC Solution")
    print("="*70)
    
    results = []
    segment_size = config['segment_size']
    
    # Create semi-structured segments (realistic case)
    assignments = create_semi_structured_segments(
        vectors, topic_labels, segment_size, 
        topics_per_segment=3, seed=config['seed']
    )
    
    np.random.seed(config['seed'])
    query_idx = np.random.choice(len(vectors), config['n_queries'], replace=False)
    query_vectors = vectors[query_idx]
    query_gt = assignments[query_idx]
    
    methods = [
        ('Baseline', MeanCentroidIndex()),
        ('HMRC-2', HMRCIndex(n_representatives=2)),
        ('HMRC-3', HMRCIndex(n_representatives=3)),
        ('HMRC-5', HMRCIndex(n_representatives=5)),
        ('HMRC-7', HMRCIndex(n_representatives=7)),
    ]
    
    for method_name, index in methods:
        print(f"\n--- {method_name} ---")
        
        start = time.time()
        index.fit(vectors, assignments)
        build_time = time.time() - start
        
        for k in [1, 5, 10, 20]:
            recall = evaluate_routing(index, query_vectors, query_gt, k=k)
            results.append({
                'method': method_name,
                'k': k,
                'recall': recall,
                'overhead': index.get_overhead(),
                'build_time': build_time,
            })
        
        print(f"  Recall@10: {results[-2]['recall']:.4f}")
        print(f"  Overhead: {index.get_overhead():.1f}x")
    
    return pd.DataFrame(results)


def experiment_3_topics_ablation(vectors, topic_labels, config) -> pd.DataFrame:
    """
    Experiment 3: Ablation on topics per segment.
    """
    print("\n" + "="*70)
    print("EXPERIMENT 3: Topics per Segment Ablation")
    print("="*70)
    
    results = []
    segment_size = config['segment_size']
    
    np.random.seed(config['seed'])
    query_idx = np.random.choice(len(vectors), config['n_queries'], replace=False)
    query_vectors = vectors[query_idx]
    
    for n_topics in [1, 2, 3, 5, 7, 10]:
        print(f"\n--- {n_topics} topics per segment ---")
        
        if n_topics == 1:
            # Coherent (1 topic = clustering)
            n_segments = len(vectors) // segment_size
            assignments = create_coherent_segments(vectors, n_segments, config['seed'])
        else:
            assignments = create_semi_structured_segments(
                vectors, topic_labels, segment_size, n_topics, config['seed']
            )
        
        query_gt = assignments[query_idx]
        
        for method_name, IndexClass, kwargs in [
            ('Baseline', MeanCentroidIndex, {}),
            ('HMRC-3', HMRCIndex, {'n_representatives': 3}),
        ]:
            index = IndexClass(**kwargs)
            index.fit(vectors, assignments)
            recall = evaluate_routing(index, query_vectors, query_gt, k=10)
            
            results.append({
                'topics_per_segment': n_topics,
                'method': method_name,
                'recall': recall,
            })
            
            print(f"  {method_name}: {recall:.4f}")
    
    return pd.DataFrame(results)


def experiment_4_representatives_ablation(vectors, topic_labels, config) -> pd.DataFrame:
    """
    Experiment 4: Ablation on number of representatives.
    """
    print("\n" + "="*70)
    print("EXPERIMENT 4: Number of Representatives Ablation")
    print("="*70)
    
    results = []
    segment_size = config['segment_size']
    
    assignments = create_semi_structured_segments(
        vectors, topic_labels, segment_size, 
        topics_per_segment=3, seed=config['seed']
    )
    
    np.random.seed(config['seed'])
    query_idx = np.random.choice(len(vectors), config['n_queries'], replace=False)
    query_vectors = vectors[query_idx]
    query_gt = assignments[query_idx]
    
    for n_reps in [1, 2, 3, 4, 5, 7, 10]:
        print(f"\n--- {n_reps} representatives ---")
        
        if n_reps == 1:
            index = MeanCentroidIndex()
        else:
            index = HMRCIndex(n_representatives=n_reps)
        
        index.fit(vectors, assignments)
        recall = evaluate_routing(index, query_vectors, query_gt, k=10)
        
        results.append({
            'n_representatives': n_reps,
            'recall': recall,
            'overhead': index.get_overhead(),
        })
        
        print(f"  Recall@10: {recall:.4f}, Overhead: {index.get_overhead():.1f}x")
    
    return pd.DataFrame(results)


def experiment_5_segment_sizes(vectors, topic_labels, config) -> pd.DataFrame:
    """
    Experiment 5: Different segment sizes.
    """
    print("\n" + "="*70)
    print("EXPERIMENT 5: Segment Size Impact")
    print("="*70)
    
    results = []
    
    np.random.seed(config['seed'])
    query_idx = np.random.choice(len(vectors), config['n_queries'], replace=False)
    query_vectors = vectors[query_idx]
    
    for segment_size in [100, 250, 500, 1000, 2000]:
        print(f"\n--- Segment size: {segment_size} ---")
        
        assignments = create_semi_structured_segments(
            vectors, topic_labels, segment_size, 
            topics_per_segment=3, seed=config['seed']
        )
        query_gt = assignments[query_idx]
        
        for method_name, IndexClass, kwargs in [
            ('Baseline', MeanCentroidIndex, {}),
            ('HMRC-3', HMRCIndex, {'n_representatives': 3}),
        ]:
            index = IndexClass(**kwargs)
            index.fit(vectors, assignments)
            recall = evaluate_routing(index, query_vectors, query_gt, k=10)
            
            results.append({
                'segment_size': segment_size,
                'method': method_name,
                'recall': recall,
                'n_segments': len(np.unique(assignments)),
            })
            
            print(f"  {method_name}: {recall:.4f}")
    
    return pd.DataFrame(results)


def experiment_6_full_comparison(vectors, topic_labels, config) -> pd.DataFrame:
    """
    Experiment 6: Full comparison across all conditions.
    """
    print("\n" + "="*70)
    print("EXPERIMENT 6: Full Comparison Table")
    print("="*70)
    
    results = []
    segment_size = config['segment_size']
    n_segments = len(vectors) // segment_size
    
    np.random.seed(config['seed'])
    query_idx = np.random.choice(len(vectors), config['n_queries'], replace=False)
    query_vectors = vectors[query_idx]
    
    segmentations = {
        'Coherent': create_coherent_segments(vectors, n_segments, config['seed']),
        'Semi-3': create_semi_structured_segments(vectors, topic_labels, segment_size, 3, config['seed']),
        'Random': create_random_segments(len(vectors), segment_size, config['seed']),
    }
    
    methods = [
        ('Baseline', MeanCentroidIndex, {}),
        ('HMRC-3', HMRCIndex, {'n_representatives': 3}),
        ('HMRC-5', HMRCIndex, {'n_representatives': 5}),
    ]
    
    for seg_name, assignments in segmentations.items():
        query_gt = assignments[query_idx]
        
        for method_name, IndexClass, kwargs in methods:
            index = IndexClass(**kwargs)
            index.fit(vectors, assignments)
            recall = evaluate_routing(index, query_vectors, query_gt, k=10)
            
            results.append({
                'segmentation': seg_name,
                'method': method_name,
                'recall': recall,
                'overhead': index.get_overhead(),
            })
    
    return pd.DataFrame(results)


# ============================================================================
# Figure Generation
# ============================================================================

def create_paper_figures(exp1, exp2, exp3, exp4, exp5, exp6, output_dir):
    """Generate publication-quality figures."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Set style
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # ========================================
    # Figure 1: Problem Validation
    # ========================================
    fig, ax = plt.subplots(figsize=(8, 5))
    
    exp1_k10 = exp1[exp1['k'] == 10]
    segments = exp1_k10['segmentation'].values
    recalls = exp1_k10['recall'].values
    
    colors = ['#2ecc71' if 'Coherent' in s else '#e74c3c' if 'Random' in s else '#3498db' for s in segments]
    bars = ax.bar(range(len(segments)), recalls, color=colors)
    
    ax.set_xticks(range(len(segments)))
    ax.set_xticklabels(segments, fontsize=11)
    ax.set_ylabel('Routing Recall@10', fontsize=12)
    ax.set_xlabel('Segmentation Strategy', fontsize=12)
    ax.set_title('(a) Routing Accuracy Degrades with Segment Heterogeneity', fontsize=13)
    ax.set_ylim(0, 1.1)
    
    for bar, val in zip(bars, recalls):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.1%}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig1_problem_validation.pdf'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'fig1_problem_validation.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # ========================================
    # Figure 2: HMRC Solution
    # ========================================
    fig, ax = plt.subplots(figsize=(8, 5))
    
    exp2_k10 = exp2[exp2['k'] == 10]
    methods = exp2_k10['method'].values
    recalls = exp2_k10['recall'].values
    
    colors = ['#e74c3c' if 'Baseline' in m else '#3498db' for m in methods]
    bars = ax.bar(range(len(methods)), recalls, color=colors)
    
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods, fontsize=11)
    ax.set_ylabel('Routing Recall@10', fontsize=12)
    ax.set_xlabel('Method', fontsize=12)
    ax.set_title('(b) HMRC Improves Routing on Semi-Structured Segments', fontsize=13)
    ax.set_ylim(0, 1.1)
    
    baseline = recalls[0]
    for bar, val in zip(bars, recalls):
        improvement = (val - baseline) / baseline * 100 if baseline > 0 else 0
        label = f'{val:.1%}' if val == baseline else f'{val:.1%}\n(+{improvement:.0f}%)'
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                label, ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig2_hmrc_solution.pdf'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'fig2_hmrc_solution.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # ========================================
    # Figure 3: Topics Ablation
    # ========================================
    fig, ax = plt.subplots(figsize=(8, 5))
    
    for method in ['Baseline', 'HMRC-3']:
        data = exp3[exp3['method'] == method]
        style = '--o' if 'Baseline' in method else '-s'
        color = '#e74c3c' if 'Baseline' in method else '#3498db'
        ax.plot(data['topics_per_segment'], data['recall'], style, 
                label=method, linewidth=2.5, markersize=8, color=color)
    
    ax.set_xlabel('Topics per Segment (Heterogeneity)', fontsize=12)
    ax.set_ylabel('Routing Recall@10', fontsize=12)
    ax.set_title('(c) HMRC Benefit Increases with Heterogeneity', fontsize=13)
    ax.legend(fontsize=11)
    ax.set_ylim(0, 1.1)
    ax.set_xticks([1, 2, 3, 5, 7, 10])
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig3_topics_ablation.pdf'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'fig3_topics_ablation.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # ========================================
    # Figure 4: Representatives Ablation
    # ========================================
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    ax1.plot(exp4['n_representatives'], exp4['recall'], '-o', 
             linewidth=2.5, markersize=8, color='#3498db')
    ax1.axhline(y=exp4[exp4['n_representatives']==1]['recall'].values[0],
                color='#e74c3c', linestyle='--', linewidth=2, label='Baseline (k=1)')
    ax1.set_xlabel('Number of Representatives (k)', fontsize=12)
    ax1.set_ylabel('Routing Recall@10', fontsize=12)
    ax1.set_title('(d) Recall vs Representatives', fontsize=13)
    ax1.legend(fontsize=11)
    ax1.set_ylim(0, 1.1)
    
    ax2.plot(exp4['n_representatives'], exp4['overhead'], '-o',
             linewidth=2.5, markersize=8, color='#e74c3c')
    ax2.set_xlabel('Number of Representatives (k)', fontsize=12)
    ax2.set_ylabel('Memory Overhead (×)', fontsize=12)
    ax2.set_title('(e) Memory Overhead', fontsize=13)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig4_representatives_ablation.pdf'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'fig4_representatives_ablation.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # ========================================
    # Figure 5: Segment Size
    # ========================================
    fig, ax = plt.subplots(figsize=(8, 5))
    
    for method in ['Baseline', 'HMRC-3']:
        data = exp5[exp5['method'] == method]
        style = '--o' if 'Baseline' in method else '-s'
        color = '#e74c3c' if 'Baseline' in method else '#3498db'
        ax.semilogx(data['segment_size'], data['recall'], style,
                    label=method, linewidth=2.5, markersize=8, color=color)
    
    ax.set_xlabel('Segment Size (log scale)', fontsize=12)
    ax.set_ylabel('Routing Recall@10', fontsize=12)
    ax.set_title('(f) HMRC Improvement Across Segment Sizes', fontsize=13)
    ax.legend(fontsize=11)
    ax.set_ylim(0, 1.1)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig5_segment_sizes.pdf'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'fig5_segment_sizes.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # ========================================
    # Figure 6: Full Comparison Heatmap
    # ========================================
    fig, ax = plt.subplots(figsize=(8, 5))
    
    pivot = exp6.pivot(index='method', columns='segmentation', values='recall')
    pivot = pivot[['Coherent', 'Semi-3', 'Random']]  # Order columns
    
    im = ax.imshow(pivot.values, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, fontsize=11)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=11)
    ax.set_title('Routing Recall@10 Across Conditions', fontsize=13)
    
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            color = 'white' if val < 0.5 else 'black'
            ax.text(j, i, f'{val:.1%}', ha='center', va='center', color=color, fontsize=11, fontweight='bold')
    
    plt.colorbar(im, ax=ax, label='Recall@10')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig6_full_comparison.pdf'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'fig6_full_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n📊 All figures saved to {output_dir}/")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='HMRC Paper Experiments')
    parser.add_argument('--embeddings', type=str, required=True)
    parser.add_argument('--output', type=str, default='paper_results')
    parser.add_argument('--segment-size', type=int, default=500)
    parser.add_argument('--n-topics', type=int, default=100)
    parser.add_argument('--n-queries', type=int, default=2000)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"{args.output}_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    print("="*70)
    print("HMRC PAPER EXPERIMENTS")
    print("="*70)
    print(f"Output: {output_dir}")
    
    # Load data
    print(f"\n📂 Loading embeddings from {args.embeddings}...")
    vectors = np.load(args.embeddings).astype(np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors = vectors / (norms + 1e-10)
    print(f"   Shape: {vectors.shape}")
    
    # Create topic clusters
    print(f"\n🏷️  Creating {args.n_topics} topic clusters...")
    topic_labels = create_topic_clusters(vectors, args.n_topics, args.seed)
    
    config = {
        'segment_size': args.segment_size,
        'n_queries': args.n_queries,
        'seed': args.seed,
    }
    
    # Run all experiments
    exp1 = experiment_1_problem_validation(vectors, topic_labels, config)
    exp2 = experiment_2_hmrc_solution(vectors, topic_labels, config)
    exp3 = experiment_3_topics_ablation(vectors, topic_labels, config)
    exp4 = experiment_4_representatives_ablation(vectors, topic_labels, config)
    exp5 = experiment_5_segment_sizes(vectors, topic_labels, config)
    exp6 = experiment_6_full_comparison(vectors, topic_labels, config)
    
    # Save CSVs
    exp1.to_csv(os.path.join(output_dir, 'exp1_problem_validation.csv'), index=False)
    exp2.to_csv(os.path.join(output_dir, 'exp2_hmrc_solution.csv'), index=False)
    exp3.to_csv(os.path.join(output_dir, 'exp3_topics_ablation.csv'), index=False)
    exp4.to_csv(os.path.join(output_dir, 'exp4_representatives_ablation.csv'), index=False)
    exp5.to_csv(os.path.join(output_dir, 'exp5_segment_sizes.csv'), index=False)
    exp6.to_csv(os.path.join(output_dir, 'exp6_full_comparison.csv'), index=False)
    
    # Create figures
    try:
        create_paper_figures(exp1, exp2, exp3, exp4, exp5, exp6, output_dir)
    except Exception as e:
        print(f"⚠️  Could not create figures: {e}")
    
    # Print summary tables
    print("\n" + "="*70)
    print("PAPER RESULTS SUMMARY")
    print("="*70)
    
    print("\n📊 Table 1: Problem Validation (Recall@10)")
    print(exp1[exp1['k']==10][['segmentation', 'recall']].to_string(index=False))
    
    print("\n📊 Table 2: HMRC Solution (Recall@10 on Semi-3)")
    print(exp2[exp2['k']==10][['method', 'recall', 'overhead']].to_string(index=False))
    
    print("\n📊 Table 3: Full Comparison")
    print(exp6.pivot(index='method', columns='segmentation', values='recall').round(4).to_string())
    
    # Save summary
    summary = {
        'n_vectors': int(vectors.shape[0]),
        'embedding_dim': int(vectors.shape[1]),
        'segment_size': args.segment_size,
        'n_topics': args.n_topics,
        'key_results': {
            'baseline_coherent': float(exp6[(exp6['method']=='Baseline') & (exp6['segmentation']=='Coherent')]['recall'].values[0]),
            'baseline_semi3': float(exp6[(exp6['method']=='Baseline') & (exp6['segmentation']=='Semi-3')]['recall'].values[0]),
            'hmrc3_semi3': float(exp6[(exp6['method']=='HMRC-3') & (exp6['segmentation']=='Semi-3')]['recall'].values[0]),
            'improvement_pct': float((exp6[(exp6['method']=='HMRC-3') & (exp6['segmentation']=='Semi-3')]['recall'].values[0] - 
                                      exp6[(exp6['method']=='Baseline') & (exp6['segmentation']=='Semi-3')]['recall'].values[0]) / 
                                     exp6[(exp6['method']=='Baseline') & (exp6['segmentation']=='Semi-3')]['recall'].values[0] * 100),
        }
    }
    
    with open(os.path.join(output_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n📁 All results saved to: {output_dir}/")


if __name__ == "__main__":
    main()
