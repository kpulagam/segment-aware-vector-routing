"""
Segmentation Strategies for HMRC Experiments

Provides different strategies for creating segments with varying
levels of heterogeneity.

Usage:
    from src.segmentation import (
        create_coherent_segments,
        create_semi_structured_segments,
        create_random_segments,
        create_topic_clusters
    )
"""

import numpy as np
from sklearn.cluster import MiniBatchKMeans


def create_topic_clusters(
    vectors: np.ndarray, 
    n_topics: int, 
    seed: int = 42
) -> np.ndarray:
    """
    Pre-cluster vectors into topic clusters using k-means.
    
    Args:
        vectors: Array of shape (n_vectors, dim)
        n_topics: Number of topic clusters to create
        seed: Random seed for reproducibility
        
    Returns:
        Array of shape (n_vectors,) with topic labels
    """
    kmeans = MiniBatchKMeans(
        n_clusters=n_topics, 
        random_state=seed, 
        batch_size=1024, 
        n_init=3
    )
    return kmeans.fit_predict(vectors)


def create_coherent_segments(
    vectors: np.ndarray, 
    n_segments: int, 
    seed: int = 42
) -> np.ndarray:
    """
    Create semantically coherent segments via clustering.
    
    Each segment contains vectors from a single semantic region.
    This represents the ideal case for centroid-based routing.
    
    Args:
        vectors: Array of shape (n_vectors, dim)
        n_segments: Number of segments to create
        seed: Random seed for reproducibility
        
    Returns:
        Array of shape (n_vectors,) with segment assignments
    """
    kmeans = MiniBatchKMeans(
        n_clusters=n_segments, 
        random_state=seed, 
        batch_size=1024, 
        n_init=3
    )
    return kmeans.fit_predict(vectors)


def create_random_segments(
    n_vectors: int, 
    segment_size: int, 
    seed: int = 42
) -> np.ndarray:
    """
    Create purely random segments (worst case).
    
    Vectors are randomly shuffled and assigned to segments.
    This represents the pathological case with no structure.
    
    Args:
        n_vectors: Total number of vectors
        segment_size: Target size for each segment
        seed: Random seed for reproducibility
        
    Returns:
        Array of shape (n_vectors,) with segment assignments
    """
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
    Create semi-structured heterogeneous segments.
    
    Each segment contains vectors from `topics_per_segment` distinct topics,
    simulating realistic scenarios like:
    - Time-based batching (few topics per time window)
    - Source-based ingestion (documents from few sources)
    - Incremental updates (related docs arrive together)
    
    Args:
        vectors: Array of shape (n_vectors, dim)
        topic_labels: Array of shape (n_vectors,) with topic assignments
        segment_size: Target size for each segment
        topics_per_segment: Number of topics to mix per segment
        seed: Random seed for reproducibility
        
    Returns:
        Array of shape (n_vectors,) with segment assignments
    """
    np.random.seed(seed)
    n_vectors = len(vectors)
    n_topics = len(np.unique(topic_labels))
    n_segments = n_vectors // segment_size
    
    # Group vectors by topic
    topic_to_vectors = {}
    for i, topic in enumerate(topic_labels):
        if topic not in topic_to_vectors:
            topic_to_vectors[topic] = []
        topic_to_vectors[topic].append(i)
    
    # Shuffle within each topic
    for topic in topic_to_vectors:
        np.random.shuffle(topic_to_vectors[topic])
    
    # Track current position in each topic
    topic_positions = {t: 0 for t in topic_to_vectors}
    
    # Assign vectors to segments
    assignments = np.full(n_vectors, -1, dtype=np.int32)
    
    for seg_id in range(n_segments):
        # Select random topics for this segment
        available_topics = list(topic_to_vectors.keys())
        selected_topics = np.random.choice(
            available_topics, 
            size=min(topics_per_segment, len(available_topics)),
            replace=False
        )
        
        # Distribute segment_size vectors among selected topics
        vectors_per_topic = segment_size // len(selected_topics)
        remainder = segment_size % len(selected_topics)
        
        for i, topic in enumerate(selected_topics):
            n_from_topic = vectors_per_topic + (1 if i < remainder else 0)
            
            # Get vectors from this topic
            topic_vectors = topic_to_vectors[topic]
            start = topic_positions[topic]
            end = min(start + n_from_topic, len(topic_vectors))
            
            for idx in topic_vectors[start:end]:
                assignments[idx] = seg_id
            
            topic_positions[topic] = end
            
            # Wrap around if exhausted
            if topic_positions[topic] >= len(topic_vectors):
                topic_positions[topic] = 0
    
    # Assign any remaining vectors
    unassigned = np.where(assignments == -1)[0]
    for idx in unassigned:
        assignments[idx] = np.random.randint(n_segments)
    
    return assignments


def create_time_based_segments(
    n_vectors: int,
    segment_size: int,
) -> np.ndarray:
    """
    Create segments based on sequential order (simulating time-based ingestion).
    
    Vectors are assigned to segments in order, simulating documents
    arriving in batches over time.
    
    Args:
        n_vectors: Total number of vectors
        segment_size: Target size for each segment
        
    Returns:
        Array of shape (n_vectors,) with segment assignments
    """
    n_segments = n_vectors // segment_size
    assignments = np.repeat(np.arange(n_segments), segment_size)[:n_vectors]
    return assignments
