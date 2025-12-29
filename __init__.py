"""
HMRC: Heterogeneous Multi-Representative Centroids

A simple, effective solution for improving vector database routing
on heterogeneous segments.
"""

from .hmrc import HMRCIndex, MeanCentroidIndex, BaselineIndex
from .segmentation import (
    create_coherent_segments,
    create_semi_structured_segments,
    create_random_segments,
    create_topic_clusters,
)
from .evaluation import (
    evaluate_routing,
    compute_segment_statistics,
    generate_queries_from_corpus,
)

__version__ = "0.1.0"
__all__ = [
    "HMRCIndex",
    "MeanCentroidIndex", 
    "BaselineIndex",
    "create_coherent_segments",
    "create_semi_structured_segments",
    "create_random_segments",
    "create_topic_clusters",
    "evaluate_routing",
    "compute_segment_statistics",
    "generate_queries_from_corpus",
]
