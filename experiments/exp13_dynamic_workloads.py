#!/usr/bin/env python3
"""
Exp 13: HMRC under dynamic workloads — streaming inserts, deletes, and
multi-tenant growth (R1: "dynamic scenarios involving updates, deletes,
tenant growth, or changing segment composition").

Scenarios
  A. append   — time-drifting topic stream fills segments append-only;
                segments seal at segment_size. Corpus grows 50%..100%.
  B. deletes  — full corpus, then topic-skewed deletions (30% removed).
  C. tenants  — segments are per-tenant; tenants have drifting topic
                profiles, grow at different rates; new tenants appear.

Maintenance policies for HMRC representatives
  static    — fit at seal time (A) / once at start (B,C), never refreshed
  periodic  — refresh changed segments every `refresh_every` steps
  drift     — refresh a segment when >20% of its content changed since fit

Baseline mean centroid is always recomputed exactly (cheap running mean),
which is the strongest version of the status quo.

Usage:
    python exp13_dynamic_workloads.py --embeddings ../data/embeddings/msmarco_100000.npy \
        --dataset-name MSMARCO-100K --output revision_results/exp13 --seeds 42 43 44
"""

import os
os.environ.setdefault('OMP_NUM_THREADS', '1')

import argparse
import json
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd

from common import (make_topic_labels, pairwise_sq_dists,
                    route_topk_segments, routing_recall, agg_mean_std)

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from sklearn.cluster import KMeans

SEGMENT_SIZE = 500
N_REPS = 3
K_ROUTE = 10
N_QUERIES = 1000
REFRESH_EVERY = 5
DRIFT_THRESHOLD = 0.20


# ---------------------------------------------------------------------------
# Index state that supports incremental maintenance
# ---------------------------------------------------------------------------

class DynamicIndex:
    """Segment-level index with mean centroids and HMRC reps + policies."""

    def __init__(self, policy: str, seed: int = 42):
        self.policy = policy
        self.seed = seed
        self.seg_members = {}        # seg_id -> list of vector row ids
        self.means = {}              # seg_id -> running mean
        self.reps = {}               # seg_id -> (r, dim) array
        self.fitted_size = {}        # size when reps last fitted
        self.changes_since_fit = {}  # inserts+deletes since last fit
        self.kmeans_fits = 0

    def _fit_reps(self, seg_id, vectors):
        member_vecs = vectors[self.seg_members[seg_id]]
        k = min(N_REPS, max(1, len(member_vecs) // 2))
        if len(member_vecs) < 10 or k == 1:
            self.reps[seg_id] = member_vecs.mean(0, keepdims=True)
        else:
            km = KMeans(n_clusters=k, random_state=self.seed, n_init=3,
                        max_iter=100).fit(member_vecs.astype(np.float64))
            self.reps[seg_id] = km.cluster_centers_.astype(np.float32)
            self.kmeans_fits += 1
        self.fitted_size[seg_id] = len(member_vecs)
        self.changes_since_fit[seg_id] = 0

    def insert(self, seg_id, row_ids, vectors):
        members = self.seg_members.setdefault(seg_id, [])
        members.extend(row_ids)
        vecs = vectors[members]
        self.means[seg_id] = vecs.mean(0)
        if seg_id not in self.reps:
            self._fit_reps(seg_id, vectors)
        else:
            self.changes_since_fit[seg_id] = (
                self.changes_since_fit.get(seg_id, 0) + len(row_ids))

    def delete(self, seg_id, row_ids, vectors):
        s = set(row_ids)
        self.seg_members[seg_id] = [r for r in self.seg_members[seg_id]
                                    if r not in s]
        if not self.seg_members[seg_id]:
            for d in (self.seg_members, self.means, self.reps,
                      self.fitted_size, self.changes_since_fit):
                d.pop(seg_id, None)
            return
        self.means[seg_id] = vectors[self.seg_members[seg_id]].mean(0)
        self.changes_since_fit[seg_id] = (
            self.changes_since_fit.get(seg_id, 0) + len(row_ids))

    def maintain(self, vectors, step):
        if self.policy == 'static':
            return
        for seg_id in list(self.seg_members):
            changed = self.changes_since_fit.get(seg_id, 0)
            if changed == 0:
                continue
            base = max(self.fitted_size.get(seg_id, 1), 1)
            if self.policy == 'periodic' and step % REFRESH_EVERY == 0:
                self._fit_reps(seg_id, vectors)
            elif self.policy == 'drift' and changed / base > DRIFT_THRESHOLD:
                self._fit_reps(seg_id, vectors)

    def routing_arrays(self, method):
        segs = sorted(self.seg_members)
        if method == 'mean':
            reps = np.vstack([self.means[s] for s in segs]).astype(np.float32)
            r2s = np.array(segs)
        else:
            reps = np.vstack([self.reps[s] for s in segs]).astype(np.float32)
            r2s = np.concatenate([[s] * len(self.reps[s]) for s in segs])
        return reps, r2s.astype(np.int64)


def eval_recall(index, method, vectors, rng, live_rows, row_to_seg):
    qidx = rng.choice(live_rows, min(N_QUERIES, len(live_rows)),
                      replace=False)
    queries = vectors[qidx]
    gt = np.array([row_to_seg[r] for r in qidx])
    reps, r2s = index.routing_arrays(method)
    routed = route_topk_segments(queries, reps, r2s, K_ROUTE)
    return routing_recall(routed, gt, K_ROUTE)


# ---------------------------------------------------------------------------
# Scenario generators: yield (step, event_type, seg_id, row_ids)
# ---------------------------------------------------------------------------

def drifting_stream(topic_labels, rng, n_steps, per_step):
    """Order rows so that topic popularity drifts over time."""
    n_topics = topic_labels.max() + 1
    by_topic = {t: list(np.where(topic_labels == t)[0]) for t in range(n_topics)}
    for t in by_topic:
        rng.shuffle(by_topic[t])
    weights = rng.dirichlet(np.ones(n_topics) * 0.5)
    order = []
    for _ in range(n_steps):
        # random-walk drift of topic mixture
        weights = np.maximum(weights + rng.normal(0, 0.02, n_topics), 1e-4)
        weights /= weights.sum()
        counts = rng.multinomial(per_step, weights)
        picked = 0
        for t in np.argsort(-counts):
            want = counts[t]
            take = min(want, len(by_topic[t]))
            for _ in range(take):
                order.append(by_topic[t].pop())
            picked += take
        # topics exhausted: backfill from whatever remains
        if picked < per_step:
            for t in range(n_topics):
                while picked < per_step and by_topic[t]:
                    order.append(by_topic[t].pop())
                    picked += 1
            if picked < per_step:
                return order
    return order


def run_scenario(args):
    scenario, policy, seed, vectors, topic_labels = args
    rng = np.random.RandomState(seed)
    n = len(vectors)
    rows = []

    if scenario == 'append':
        n_steps = 20
        per_step = (n // 2) // n_steps
        stream = drifting_stream(topic_labels, rng, n_steps * 2, per_step)
        index = DynamicIndex(policy, seed)
        row_to_seg = {}
        open_seg, seg_fill = 0, 0
        t0 = time.time()
        consumed = 0
        for step in range(n_steps * 2):
            batch = stream[consumed:consumed + per_step]
            consumed += per_step
            i = 0
            while i < len(batch):
                space = SEGMENT_SIZE - seg_fill
                chunk = list(batch[i:i + space])
                index.insert(open_seg, chunk, vectors)
                for r in chunk:
                    row_to_seg[r] = open_seg
                seg_fill += len(chunk)
                i += len(chunk)
                if seg_fill >= SEGMENT_SIZE:
                    open_seg += 1
                    seg_fill = 0
            index.maintain(vectors, step)
            if step >= n_steps and step % 2 == 0:  # measure in second half
                live = [r for r in row_to_seg]
                for method in (['mean'] if policy == 'static' else []) + ['hmrc']:
                    rec = eval_recall(index, method, vectors, rng,
                                      np.array(live), row_to_seg)
                    rows.append({'scenario': scenario, 'policy': policy,
                                 'method': method, 'seed': seed, 'step': step,
                                 'corpus_size': len(live),
                                 'recall@10': rec,
                                 'kmeans_fits': index.kmeans_fits,
                                 'wall_s': time.time() - t0})
        return rows

    if scenario == 'deletes':
        n_steps = 15
        index = DynamicIndex(policy, seed)
        row_to_seg = {}
        from common import make_segments
        assignments = make_segments(vectors, topic_labels, 'semi',
                                    SEGMENT_SIZE, seed=seed)
        for seg_id in np.unique(assignments):
            batch = np.where(assignments == seg_id)[0]
            index.insert(int(seg_id), list(batch), vectors)
            for r in batch:
                row_to_seg[r] = int(seg_id)
        # re-fit all reps once at start (static baseline state)
        live = set(row_to_seg)
        # topic-skewed deletions: half the topics face 5x delete pressure
        n_topics = topic_labels.max() + 1
        hot = set(rng.choice(n_topics, n_topics // 2, replace=False))
        del_w = np.array([5.0 if topic_labels[r] in hot else 1.0
                          for r in range(n)])
        t0 = time.time()
        for step in range(n_steps):
            live_arr = np.fromiter(live, dtype=np.int64)
            w = del_w[live_arr]
            w = w / w.sum()
            n_del = int(0.02 * len(live_arr))
            dels = rng.choice(live_arr, n_del, replace=False, p=w)
            by_seg = {}
            for r in dels:
                by_seg.setdefault(row_to_seg[r], []).append(r)
                live.discard(r)
            for seg_id, rs in by_seg.items():
                index.delete(seg_id, rs, vectors)
            index.maintain(vectors, step)
            if step % 3 == 2:
                live_arr = np.fromiter(live, dtype=np.int64)
                for method in (['mean'] if policy == 'static' else []) + ['hmrc']:
                    rec = eval_recall(index, method, vectors, rng, live_arr,
                                      row_to_seg)
                    rows.append({'scenario': scenario, 'policy': policy,
                                 'method': method, 'seed': seed, 'step': step,
                                 'corpus_size': len(live_arr),
                                 'recall@10': rec,
                                 'kmeans_fits': index.kmeans_fits,
                                 'wall_s': time.time() - t0})
        return rows

    if scenario == 'tenants':
        n_steps = 20
        n_tenants0, max_new = 40, 20
        n_topics = topic_labels.max() + 1
        by_topic = {t: list(np.where(topic_labels == t)[0])
                    for t in range(n_topics)}
        for t in by_topic:
            rng.shuffle(by_topic[t])
        index = DynamicIndex(policy, seed)
        row_to_seg = {}
        tenants = {}
        def new_tenant(tid):
            profile = rng.choice(n_topics, 3, replace=False)
            tenants[tid] = {'profile': list(profile),
                            'rate': int(rng.gamma(2.0, 40) + 10)}
        for tid in range(n_tenants0):
            new_tenant(tid)
        t0 = time.time()
        for step in range(n_steps):
            if step > 0 and step % 4 == 0 and len(tenants) < n_tenants0 + max_new:
                new_tenant(len(tenants))
            for tid, t_info in tenants.items():
                got = []
                topic_picks = rng.choice(t_info['profile'], t_info['rate'])
                for tp in topic_picks:
                    if by_topic[tp]:
                        got.append(by_topic[tp].pop())
                if not got:
                    continue
                # tenant drift: occasionally swap one profile topic
                if rng.rand() < 0.05:
                    t_info['profile'][rng.randint(3)] = rng.randint(n_topics)
                for r in got:
                    row_to_seg[r] = tid
                index.insert(tid, got, vectors)
            index.maintain(vectors, step)
            if step % 4 == 3:
                live_arr = np.fromiter(row_to_seg, dtype=np.int64)
                for method in (['mean'] if policy == 'static' else []) + ['hmrc']:
                    rec = eval_recall(index, method, vectors, rng, live_arr,
                                      row_to_seg)
                    rows.append({'scenario': scenario, 'policy': policy,
                                 'method': method, 'seed': seed, 'step': step,
                                 'corpus_size': len(live_arr),
                                 'n_tenants': len(tenants),
                                 'recall@10': rec,
                                 'kmeans_fits': index.kmeans_fits,
                                 'wall_s': time.time() - t0})
        return rows

    raise ValueError(scenario)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--embeddings', required=True)
    ap.add_argument('--dataset-name', default='MSMARCO-100K')
    ap.add_argument('--output', default='revision_results/exp13')
    ap.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44])
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--n-topics', type=int, default=100)
    args = ap.parse_args()

    vectors = np.load(args.embeddings).astype(np.float32)
    topic_labels = make_topic_labels(vectors, args.n_topics, seed=42)

    jobs = [(sc, pol, sd, vectors, topic_labels)
            for sc in ['append', 'deletes', 'tenants']
            for pol in ['static', 'periodic', 'drift']
            for sd in args.seeds]
    if args.workers > 1:
        with Pool(args.workers) as pool:
            results = pool.map(run_scenario, jobs)
    else:
        results = [run_scenario(j) for j in jobs]

    df = pd.DataFrame([r for rows in results for r in rows])
    out = f"{args.output}_{args.dataset_name}"
    os.makedirs(out, exist_ok=True)
    df.to_csv(os.path.join(out, 'exp13_raw.csv'), index=False)
    agg = agg_mean_std(df, ['scenario', 'policy', 'method', 'step'],
                       ['recall@10', 'kmeans_fits'])
    agg.to_csv(os.path.join(out, 'exp13_aggregated.csv'), index=False)
    summary = agg_mean_std(df, ['scenario', 'policy', 'method'],
                           ['recall@10', 'kmeans_fits'])
    summary.to_csv(os.path.join(out, 'exp13_summary.csv'), index=False)
    print(summary.to_string(index=False))


if __name__ == '__main__':
    main()
