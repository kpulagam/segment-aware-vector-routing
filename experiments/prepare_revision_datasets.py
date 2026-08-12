#!/usr/bin/env python3
"""
Prepare all datasets needed for the IEEE Access revision experiments.
Run on a machine with internet + sentence-transformers (GPU/MPS auto-used).

Tasks
  nq        NQ-100K-v2: 100K NQ passages INCLUDING all test-qrels docs,
            with doc ids, real test-query embeddings, and qrels.
  msmarco   MSMARCO-100K-v2: same construction from BEIR msmarco with
            dev queries (streams corpus.jsonl; needs ~4GB disk).
  mdm       MDM-100K multi-domain mix: SciFact + NFCorpus + FiQA + ArguAna
            + SciDocs (~100K docs, 5 real domains) + all test queries/qrels.
  scale     MSMARCO-1M: 1M passages (incl. dev-qrels docs) for exp15.
  second    Re-embed NQ-100K-v2 + MDM-100K with BAAI/bge-base-en-v1.5.

Usage
  pip install sentence-transformers beir tqdm
  python prepare_revision_datasets.py --task nq
  python prepare_revision_datasets.py --task mdm
  python prepare_revision_datasets.py --task msmarco
  python prepare_revision_datasets.py --task scale
  python prepare_revision_datasets.py --task second

All outputs land in experiments/data/revision/. Embedding is chunked and
resumable: re-running a task continues from the last finished chunk.
"""

import argparse
import json
import os
import zipfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'data', 'revision')
BEIR_DIR = os.path.join(HERE, 'data', 'beir')
BEIR_URL = ('https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/'
            'datasets/{}.zip')
MINILM = 'sentence-transformers/all-MiniLM-L6-v2'
BGE = 'BAAI/bge-base-en-v1.5'
BGE_QUERY_PREFIX = ('Represent this sentence for searching relevant '
                    'passages: ')
CHUNK = 10000


def get_device():
    import torch
    if torch.cuda.is_available():
        return 'cuda'
    if getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


def get_model(name):
    from sentence_transformers import SentenceTransformer
    dev = get_device()
    print(f'Loading {name} on {dev}')
    return SentenceTransformer(name, device=dev)


def embed_resumable(texts, model, tag, batch_size=128):
    """Chunked, resumable embedding. Saves parts under OUT/parts/<tag>/."""
    part_dir = os.path.join(OUT, 'parts', tag)
    os.makedirs(part_dir, exist_ok=True)
    n_chunks = (len(texts) + CHUNK - 1) // CHUNK
    for ci in range(n_chunks):
        pf = os.path.join(part_dir, f'{ci:05d}.npy')
        if os.path.exists(pf):
            continue
        chunk = texts[ci * CHUNK:(ci + 1) * CHUNK]
        embs = model.encode(chunk, batch_size=batch_size,
                            show_progress_bar=True, convert_to_numpy=True,
                            normalize_embeddings=False)
        np.save(pf, embs.astype(np.float32))
        print(f'{tag}: chunk {ci + 1}/{n_chunks} done')
    return np.vstack([np.load(os.path.join(part_dir, f'{ci:05d}.npy'))
                      for ci in range(n_chunks)])


def download_beir(dataset, max_retries=5):
    import time
    import urllib.request
    path = os.path.join(BEIR_DIR, dataset)
    if os.path.isdir(path) and os.path.exists(os.path.join(path, 'corpus.jsonl')):
        return path
    os.makedirs(BEIR_DIR, exist_ok=True)
    zpath = os.path.join(BEIR_DIR, f'{dataset}.zip')
    url = BEIR_URL.format(dataset)
    for attempt in range(1, max_retries + 1):
        # validate any existing zip (may be a truncated leftover)
        if os.path.exists(zpath):
            try:
                with zipfile.ZipFile(zpath) as z:
                    if z.testzip() is None:
                        break  # good zip
            except zipfile.BadZipFile:
                pass
            print(f'{dataset}.zip is incomplete/corrupt, deleting and retrying')
            os.remove(zpath)
        try:
            print(f'Downloading {url} (attempt {attempt}/{max_retries})')
            urllib.request.urlretrieve(url, zpath + '.part')
            os.rename(zpath + '.part', zpath)
        except Exception as e:
            print(f'  download failed: {e}')
            for f in (zpath + '.part', zpath):
                if os.path.exists(f):
                    os.remove(f)
            if attempt == max_retries:
                raise
            time.sleep(5 * attempt)
    with zipfile.ZipFile(zpath) as z:
        z.extractall(BEIR_DIR)
    return path


def iter_corpus(path):
    with open(os.path.join(path, 'corpus.jsonl')) as f:
        for line in f:
            d = json.loads(line)
            title = d.get('title', '') or ''
            text = d.get('text', '') or ''
            yield d['_id'], (f'{title} {text}'.strip() if title else text)


def load_queries(path):
    out = {}
    with open(os.path.join(path, 'queries.jsonl')) as f:
        for line in f:
            d = json.loads(line)
            out[d['_id']] = d['text']
    return out


def load_qrels(path, split):
    qrels = {}
    fp = os.path.join(path, 'qrels', f'{split}.tsv')
    with open(fp) as f:
        next(f)  # header
        for line in f:
            qid, did, score = line.strip().split('\t')
            if int(score) > 0:
                qrels.setdefault(qid, set()).add(did)
    return qrels


def build_subset(beir_path, split, n_docs, seed, tag, model,
                 query_prefix='', batch_size=128):
    """Corpus subset containing all qrels docs + random fill to n_docs."""
    qrels = load_qrels(beir_path, split)
    queries = load_queries(beir_path)
    needed = set()
    for dids in qrels.values():
        needed.update(dids)
    # NOTE: doc order changed (2026-07 fix). If a bundle for this tag was
    # embedded before the fix, its cached chunks under data/revision/parts/
    # are in the OLD order — delete parts/<tag>_docs before re-running, or
    # use fix_bundle_order.py on the existing .npz instead (no re-embed).
    # pass 1: enumerate ids; select qrels docs + random fill
    all_ids = [did for did, _ in iter_corpus(beir_path)]
    present_needed = set(d for d in all_ids if d in needed)
    n_fill = n_docs - len(present_needed)
    rng = np.random.RandomState(seed)
    pool = [d for d in all_ids if d not in present_needed]
    fill = set(np.array(pool)[rng.choice(len(pool), min(n_fill, len(pool)),
                                         replace=False)])
    chosen = present_needed | fill
    # pass 2: keep docs in ORIGINAL corpus order (so 'timebatch'
    # segmentation reflects ingestion order, not construction order)
    keep_ids, keep_texts = [], []
    for did, text in iter_corpus(beir_path):
        if did in chosen:
            keep_ids.append(did)
            keep_texts.append(text)
    print(f'{tag}: {len(keep_ids)} docs ({len(present_needed)} qrels-required)')
    doc_embs = embed_resumable(keep_texts, model, f'{tag}_docs', batch_size)
    qids = [q for q in qrels if q in queries]
    q_texts = [query_prefix + queries[q] for q in qids]
    q_embs = embed_resumable(q_texts, model, f'{tag}_queries', batch_size)
    # qrels as flat pairs
    kept = set(keep_ids)
    pairs = [(qi, did) for qi, qid in enumerate(qids)
             for did in qrels[qid] if did in kept]
    np.savez_compressed(
        os.path.join(OUT, f'{tag}.npz'),
        doc_embs=doc_embs, doc_ids=np.array(keep_ids),
        query_embs=q_embs, query_ids=np.array(qids),
        qrel_query_idx=np.array([p[0] for p in pairs]),
        qrel_doc_id=np.array([p[1] for p in pairs]))
    print(f'saved {tag}.npz')


def task_nq(model_name=MINILM, tag='nq100k_v2', prefix=''):
    path = download_beir('nq')
    model = get_model(model_name)
    build_subset(path, 'test', 100000, 42, tag, model, prefix)


def task_msmarco():
    path = download_beir('msmarco')
    model = get_model(MINILM)
    build_subset(path, 'dev', 100000, 42, 'msmarco100k_v2', model)


MDM_DATASETS = ['scifact', 'nfcorpus', 'fiqa', 'arguana', 'scidocs']
MDM_SPLITS = {'scifact': 'test', 'nfcorpus': 'test', 'fiqa': 'test',
              'arguana': 'test', 'scidocs': 'test'}


def task_mdm(model_name=MINILM, tag='mdm100k', prefix=''):
    model = get_model(model_name)
    texts, doc_ids, domains = [], [], []
    all_queries, all_qrels = [], []  # (domain, qid, text) / (qid, did)
    for dom_idx, ds in enumerate(MDM_DATASETS):
        path = download_beir(ds)
        for did, text in iter_corpus(path):
            doc_ids.append(f'{ds}:{did}')
            texts.append(text)
            domains.append(dom_idx)
        queries = load_queries(path)
        qrels = load_qrels(path, MDM_SPLITS[ds])
        for qid, dids in qrels.items():
            if qid not in queries:
                continue
            all_queries.append((dom_idx, f'{ds}:{qid}',
                                prefix + queries[qid]))
            for did in dids:
                all_qrels.append((f'{ds}:{qid}', f'{ds}:{did}'))
        print(f'{ds}: corpus so far {len(texts)}')
    doc_embs = embed_resumable(texts, model, f'{tag}_docs')
    q_embs = embed_resumable([q[2] for q in all_queries], model,
                             f'{tag}_queries')
    qid_to_idx = {q[1]: i for i, q in enumerate(all_queries)}
    np.savez_compressed(
        os.path.join(OUT, f'{tag}.npz'),
        doc_embs=doc_embs, doc_ids=np.array(doc_ids),
        domains=np.array(domains),
        query_embs=q_embs,
        query_ids=np.array([q[1] for q in all_queries]),
        query_domains=np.array([q[0] for q in all_queries]),
        qrel_query_idx=np.array([qid_to_idx[q] for q, d in all_qrels]),
        qrel_doc_id=np.array([d for q, d in all_qrels]))
    print(f'saved {tag}.npz  docs={len(texts)} queries={len(all_queries)}')


def task_scale():
    path = download_beir('msmarco')
    model = get_model(MINILM)
    qrels = load_qrels(path, 'dev')
    needed = set()
    for dids in qrels.values():
        needed.update(dids)
    n_docs = 1000000
    all_ids = [did for did, _ in iter_corpus(path)]
    rng = np.random.RandomState(42)
    keep_set = set(d for d in all_ids if d in needed)
    pool = [d for d in all_ids if d not in keep_set]
    fill = set(np.array(pool)[rng.choice(len(pool), n_docs - len(keep_set),
                                         replace=False)])
    chosen = keep_set | fill
    texts, ids = [], []
    for did, text in iter_corpus(path):
        if did in chosen:
            ids.append(did)
            texts.append(text)
    embs = embed_resumable(texts, model, 'msmarco1m_docs')
    np.save(os.path.join(OUT, 'msmarco_1m.npy'), embs)
    np.save(os.path.join(OUT, 'msmarco_1m_ids.npy'), np.array(ids))
    print('saved msmarco_1m.npy', embs.shape)


def task_second():
    task_nq(BGE, 'nq100k_v2_bge', BGE_QUERY_PREFIX)
    task_mdm(BGE, 'mdm100k_bge', BGE_QUERY_PREFIX)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--task', required=True,
                    choices=['nq', 'msmarco', 'mdm', 'scale', 'second'])
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    {'nq': task_nq, 'msmarco': task_msmarco, 'mdm': task_mdm,
     'scale': task_scale, 'second': task_second}[args.task]()
