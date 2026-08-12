#!/usr/bin/env python3
"""
Fix document order in v2 bundles.

build_subset() originally wrote all qrels-required docs first, then the
random fill. Under 'timebatch' segmentation (row order = ingestion order)
this packs every relevant document into the first few segments, corrupting
the real-query timebatch evaluation. This script permutes doc_embs/doc_ids
back to the original corpus.jsonl order (a faithful ingestion-order proxy,
matching the paper's exp8 protocol). No re-embedding needed; semi/domain
results are order-independent and unaffected.

Usage:
    python fix_bundle_order.py --bundle data/revision/nq100k_v2.npz \
        --corpus data/beir/nq/corpus.jsonl
    python fix_bundle_order.py --bundle data/revision/nq100k_v2_bge.npz \
        --corpus data/beir/nq/corpus.jsonl
    python fix_bundle_order.py --bundle data/revision/msmarco100k_v2.npz \
        --corpus data/beir/msmarco/corpus.jsonl

Then re-run exp14 for the timebatch strategy only, e.g.:
    python exp14_real_queries.py --bundle data/revision/nq100k_v2.npz \
        --dataset-name NQ-100K-v2 --strategies timebatch
(the append-safe CSV replaces the old timebatch rows; semi rows persist)
"""

import argparse
import json

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bundle', required=True)
    ap.add_argument('--corpus', required=True)
    args = ap.parse_args()

    z = np.load(args.bundle, allow_pickle=False)
    data = {k: z[k] for k in z.files}
    if data.get('order_fixed', np.array(False)).item():
        print('bundle already fixed; nothing to do')
        return

    print('reading corpus order from', args.corpus)
    line_no = {}
    with open(args.corpus) as f:
        for i, line in enumerate(f):
            line_no[json.loads(line)['_id']] = i

    ids = data['doc_ids']
    keys = np.array([line_no[d] for d in ids])
    perm = np.argsort(keys, kind='stable')
    data['doc_embs'] = data['doc_embs'][perm]
    data['doc_ids'] = ids[perm]
    data['order_fixed'] = np.array(True)

    np.savez_compressed(args.bundle, **data)
    # sanity: qrels docs should now be spread through the corpus
    qrel_docs = set(data['qrel_doc_id'])
    n = len(data['doc_ids'])
    firsthalf = sum(1 for d in data['doc_ids'][:n // 2] if d in qrel_docs)
    print(f'fixed. qrels docs in first half: {firsthalf} / {len(qrel_docs)} '
          f'(should be roughly half)')


if __name__ == '__main__':
    main()
