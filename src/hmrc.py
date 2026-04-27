"""
HMRC: Heterogeneous Multi-Representative Centroids

Core implementation of the HMRC algorithm for improved vector database routing.

Usage:
    from src.hmrc import HMRCIndex, MeanCentroidIndex
    
    # Baseline
    baseline = MeanCentroidIndex()
    baseline.fit(vectors, assignments)
    routes = baseline.route(queries, k=10)
    
    # HMRC
    hmrc = HMRCIndex(n_representatives=3)
    hmrc.fit(vectors, assignments)
    routes = hmrc.route(queries, k=10)
"""

import warnings
from typing import Dict, Optional

import numpy as np
from sklearn.cluster import KMeans


class MeanCentroidIndex:
    """
    Baseline: Single mean centroid per segment.
    
    This is the standard approach used in IVF-based vector databases.
    Each segment is represented by the mean of its vectors.
    """
    
    def __init__(self):
        self.centroids: Optional[np.ndarray] = None
        self.segment_ids: Optional[np.ndarray] = None
        self.n_segments: int = 0
    
    def fit(self, vectors: np.ndarray, assignments: np.ndarray) -> 'MeanCentroidIndex':
        """
        Build index by computing mean centroid for each segment.
        
        Args:
            vectors: Array of shape (n_vectors, dim)
            assignments: Array of shape (n_vectors,) with segment IDs
            
        Returns:
            self
        """
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
        self.n_segments = len(segment_ids)
        return self
    
    def route(self, queries: np.ndarray, k: int = 10) -> np.ndarray:
        """
        Route queries to top-k segments.
        
        Args:
            queries: Array of shape (n_queries, dim)
            k: Number of segments to route to
            
        Returns:
            Array of shape (n_queries, k) with segment IDs
        """
        queries = np.asarray(queries, dtype=np.float32)
        
        # Compute distances to all centroids
        dists = np.linalg.norm(
            queries[:, None, :] - self.centroids[None, :, :], 
            axis=2
        )
        
        # Get top-k nearest centroids
        k = min(k, len(self.segment_ids))
        top_k_indices = np.argsort(dists, axis=1)[:, :k]
        
        return self.segment_ids[top_k_indices]
    
    def get_stats(self) -> Dict:
        """Return index statistics."""
        return {
            'n_segments': self.n_segments,
            'n_representatives': self.n_segments,
            'memory_overhead': 1.0,
            'method': 'MeanCentroid'
        }


class HMRCIndex:
    """
    HMRC: Heterogeneous Multi-Representative Centroids.
    
    Uses k-means clustering within each segment to generate multiple
    representative centroids that capture internal topic structure.
    
    Args:
        n_representatives: Number of centroids per segment (default: 3)
        min_segment_size: Minimum segment size for multi-rep (default: 10)
    """
    
    def __init__(self, n_representatives: int = 3, min_segment_size: int = 10):
        self.n_representatives = n_representatives
        self.min_segment_size = min_segment_size
        self.all_representatives: Optional[np.ndarray] = None
        self.rep_to_segment: Optional[np.ndarray] = None
        self.segment_ids: Optional[np.ndarray] = None
        self.n_segments: int = 0
    
    def fit(self, vectors: np.ndarray, assignments: np.ndarray) -> 'HMRCIndex':
        """
        Build index by computing k-means centroids for each segment.
        
        Args:
            vectors: Array of shape (n_vectors, dim)
            assignments: Array of shape (n_vectors,) with segment IDs
            
        Returns:
            self
        """
        unique_segments = np.unique(assignments)
        all_reps = []
        rep_to_seg = []
        
        for seg_id in unique_segments:
            mask = assignments == seg_id
            seg_vectors = vectors[mask].astype(np.float64)  # Higher precision for k-means
            n_vectors = len(seg_vectors)
            
            # Small segments: use mean
            if n_vectors < self.min_segment_size:
                reps = [seg_vectors.mean(axis=0).astype(np.float32)]
            else:
                # Determine number of representatives
                k = min(self.n_representatives, n_vectors // 2)
                k = max(k, 1)
                
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        kmeans = KMeans(
                            n_clusters=k, 
                            random_state=42, 
                            n_init=3, 
                            max_iter=100
                        )
                        kmeans.fit(seg_vectors)
                        reps = [c.astype(np.float32) for c in kmeans.cluster_centers_]
                except Exception:
                    # Fallback to mean on any error
                    reps = [seg_vectors.mean(axis=0).astype(np.float32)]
            
            all_reps.extend(reps)
            rep_to_seg.extend([seg_id] * len(reps))
        
        self.all_representatives = np.array(all_reps, dtype=np.float32)
        self.rep_to_segment = np.array(rep_to_seg)
        self.segment_ids = unique_segments
        self.n_segments = len(unique_segments)
        return self
    
    def route(self, queries: np.ndarray, k: int = 10) -> np.ndarray:
        """
        Route queries to top-k segments.
        
        Finds nearest representatives, then deduplicates by segment.
        
        Args:
            queries: Array of shape (n_queries, dim)
            k: Number of segments to route to
            
        Returns:
            Array of shape (n_queries, k) with segment IDs
        """
        queries = np.asarray(queries, dtype=np.float32)
        n_queries = len(queries)
        
        # Compute distances to all representatives
        dists = np.linalg.norm(
            queries[:, None, :] - self.all_representatives[None, :, :], 
            axis=2
        )
        
        # Get more candidates than needed (to handle deduplication)
        n_search = min(k * self.n_representatives * 2, len(self.all_representatives))
        top_indices = np.argsort(dists, axis=1)[:, :n_search]
        
        # Deduplicate by segment
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
            # Fill remaining with -1 if not enough segments
            while j < k:
                result[i, j] = -1
                j += 1
        
        return result
    
    def get_stats(self) -> Dict:
        """Return index statistics."""
        avg_reps = len(self.all_representatives) / self.n_segments if self.n_segments > 0 else 0
        return {
            'n_segments': self.n_segments,
            'n_representatives': len(self.all_representatives),
            'avg_reps_per_segment': avg_reps,
            'memory_overhead': avg_reps,
            'method': f'HMRC-{self.n_representatives}'
        }


# Alias for backward compatibility
BaselineIndex = MeanCentroidIndex
