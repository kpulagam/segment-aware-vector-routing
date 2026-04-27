"""
Evaluation Utilities for HMRC Experiments

Provides functions for evaluating routing accuracy and computing
segment statistics.

Usage:
    from src.evaluation import evaluate_routing, compute_segment_statistics
"""

from typing import Dict, List, Union
import numpy as np


def evaluate_routing(
    index,
    queries: np.ndarray,
    gt_segments: np.ndarray,
    k_values: Union[int, List[int]] = 10
) -> Union[float, Dict[int, float]]:
    """
    Evaluate routing recall at given k value(s).
    
    Routing recall@k is the fraction of queries where the ground-truth
    segment is among the top-k routed segments.
    
    Args:
        index: Index with route(queries, k) method
        queries: Array of shape (n_queries, dim)
        gt_segments: Array of shape (n_queries,) with ground-truth segment IDs
        k_values: Single k value or list of k values
        
    Returns:
        If k_values is int: single recall value
        If k_values is list: dict mapping k -> recall
    """
    # Handle single k value
    if isinstance(k_values, int):
        routed = index.route(queries, k=k_values)
        hits = sum(1 for i, gt in enumerate(gt_segments) if gt in routed[i])
        return hits / len(gt_segments)
    
    # Handle multiple k values
    max_k = max(k_values)
    routed = index.route(queries, k=max_k)
    
    recall = {}
    for k in k_values:
        hits = sum(1 for i, gt in enumerate(gt_segments) if gt in routed[i, :k])
        recall[k] = hits / len(gt_segments)
    
    return recall


def compute_segment_statistics(
    vectors: np.ndarray, 
    assignments: np.ndarray
) -> Dict:
    """
    Compute statistics about segment coherence.
    
    Args:
        vectors: Array of shape (n_vectors, dim)
        assignments: Array of shape (n_vectors,) with segment IDs
        
    Returns:
        Dictionary with statistics:
        - n_segments: Number of segments
        - mean_segment_size: Average segment size
        - mean_intra_dist: Mean distance to centroid within segments
        - std_intra_dist: Std of intra-segment distances
        - mean_variance: Mean variance within segments
    """
    unique_segments = np.unique(assignments)
    
    sizes = []
    intra_dists = []
    variances = []
    
    for seg_id in unique_segments:
        mask = assignments == seg_id
        seg_vectors = vectors[mask]
        
        sizes.append(len(seg_vectors))
        
        if len(seg_vectors) > 1:
            centroid = seg_vectors.mean(axis=0)
            dists = np.linalg.norm(seg_vectors - centroid, axis=1)
            intra_dists.append(dists.mean())
            variances.append(np.var(seg_vectors, axis=0).sum())
    
    return {
        'n_segments': len(unique_segments),
        'mean_segment_size': np.mean(sizes),
        'std_segment_size': np.std(sizes),
        'mean_intra_dist': np.mean(intra_dists) if intra_dists else 0,
        'std_intra_dist': np.std(intra_dists) if intra_dists else 0,
        'mean_variance': np.mean(variances) if variances else 0,
    }


def compute_centroid_quality(
    vectors: np.ndarray,
    assignments: np.ndarray
) -> Dict:
    """
    Compute centroid quality metrics.
    
    Measures how well centroids represent their segments.
    
    Args:
        vectors: Array of shape (n_vectors, dim)
        assignments: Array of shape (n_vectors,) with segment IDs
        
    Returns:
        Dictionary with quality metrics
    """
    unique_segments = np.unique(assignments)
    
    coverage_ratios = []
    max_dists = []
    
    for seg_id in unique_segments:
        mask = assignments == seg_id
        seg_vectors = vectors[mask]
        
        if len(seg_vectors) > 1:
            centroid = seg_vectors.mean(axis=0)
            dists = np.linalg.norm(seg_vectors - centroid, axis=1)
            
            # Coverage ratio: what fraction of vectors are "close" to centroid
            threshold = np.median(dists)
            coverage = (dists <= threshold).mean()
            coverage_ratios.append(coverage)
            
            max_dists.append(dists.max())
    
    return {
        'mean_coverage': np.mean(coverage_ratios) if coverage_ratios else 0,
        'mean_max_dist': np.mean(max_dists) if max_dists else 0,
        'max_max_dist': np.max(max_dists) if max_dists else 0,
    }


def generate_queries_from_corpus(
    vectors: np.ndarray,
    assignments: np.ndarray,
    n_queries: int,
    seed: int = 42
) -> tuple:
    """
    Generate queries by sampling from the corpus.
    
    Args:
        vectors: Array of shape (n_vectors, dim)
        assignments: Array of shape (n_vectors,) with segment IDs
        n_queries: Number of queries to generate
        seed: Random seed
        
    Returns:
        Tuple of (query_vectors, query_gt_segments)
    """
    np.random.seed(seed)
    n_queries = min(n_queries, len(vectors))
    query_idx = np.random.choice(len(vectors), n_queries, replace=False)
    
    query_vectors = vectors[query_idx]
    query_gt_segments = assignments[query_idx]
    
    return query_vectors, query_gt_segments
