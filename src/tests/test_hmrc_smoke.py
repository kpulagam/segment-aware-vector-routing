"""Smoke tests: verifies HMRC-3 beats baseline on synthetic Semi-3 segments."""
import numpy as np
from src.hmrc import HMRCIndex, MeanCentroidIndex


def test_hmrc_beats_baseline_on_heterogeneous_segments():
    rng = np.random.default_rng(42)
    d = 64
    n_topics = 9
    vectors_per_topic = 200

    topic_means = rng.normal(0, 5, size=(n_topics, d))
    vectors = np.vstack([
        topic_means[t] + rng.normal(0, 1, size=(vectors_per_topic, d))
        for t in range(n_topics)
    ]).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)

    # 3 segments, each mixing 3 topics (Semi-3)
    assignments = np.zeros(len(vectors), dtype=np.int64)
    topic_to_segment = {0: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1, 6: 2, 7: 2, 8: 2}
    for t in range(n_topics):
        start, end = t * vectors_per_topic, (t + 1) * vectors_per_topic
        assignments[start:end] = topic_to_segment[t]

    query_idx = rng.choice(len(vectors), 100, replace=False)
    queries = vectors[query_idx]
    truth = assignments[query_idx]

    baseline = MeanCentroidIndex().fit(vectors, assignments)
    hmrc = HMRCIndex(n_representatives=3).fit(vectors, assignments)

    baseline_recall = np.mean(baseline.route(queries, k=1)[:, 0] == truth)
    hmrc_recall = np.mean(hmrc.route(queries, k=1)[:, 0] == truth)

    assert hmrc_recall >= baseline_recall, \
        f"HMRC ({hmrc_recall:.2%}) should match or beat baseline ({baseline_recall:.2%})"