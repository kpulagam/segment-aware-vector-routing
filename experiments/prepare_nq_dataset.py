#!/usr/bin/env python3
"""
Prepare NQ (Natural Questions) Dataset for HMRC Experiments
============================================================

Downloads NQ corpus from BEIR, embeds with all-MiniLM-L6-v2,
and saves as numpy array compatible with run_paper_experiments.py.

Usage:
    pip install beir sentence-transformers
    python prepare_nq_dataset.py --output data/embeddings/nq_dataset.npy

Then run experiments:
    python run_paper_experiments.py --embeddings data/embeddings/nq_dataset.npy --dataset-name NQ
"""

import argparse
import os
import numpy as np


def main():
    parser = argparse.ArgumentParser(description='Prepare NQ dataset')
    parser.add_argument('--output', type=str, default='data/embeddings/nq_dataset.npy')
    parser.add_argument('--max-passages', type=int, default=100000,
                        help='Max passages to embed (NQ has ~2.68M, we sample)')
    parser.add_argument('--model', type=str, default='all-MiniLM-L6-v2')
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    
    # --- Option A: Use BEIR to download ---
    try:
        from beir import util
        from beir.datasets.data_loader import GenericDataLoader
        
        print("Downloading NQ dataset from BEIR...")
        dataset = "nq"
        url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset}.zip"
        data_path = util.download_and_unzip(url, "data/beir")
        
        corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="tests")
        
        print(f"Corpus size: {len(corpus)}")
        texts = []
        for doc_id in corpus:
            title = corpus[doc_id].get("title", "")
            text = corpus[doc_id].get("text", "")
            combined = f"{title} {text}".strip() if title else text
            texts.append(combined)
        
    except ImportError:
        # --- Option B: Manual download fallback ---
        print("BEIR not installed. Trying manual approach...")
        print("Please install: pip install beir")
        print("")
        print("Or manually download NQ from:")
        print("  https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/nq.zip")
        print("Unzip and provide the corpus.jsonl path.")
        
        import json
        corpus_path = input("Path to corpus.jsonl: ").strip()
        texts = []
        with open(corpus_path, 'r') as f:
            for line in f:
                doc = json.loads(line)
                title = doc.get("title", "")
                text = doc.get("text", "")
                combined = f"{title} {text}".strip() if title else text
                texts.append(combined)
    
    # Sample if too large
    if len(texts) > args.max_passages:
        np.random.seed(args.seed)
        indices = np.random.choice(len(texts), args.max_passages, replace=False)
        indices.sort()
        texts = [texts[i] for i in indices]
        print(f"Sampled {args.max_passages} passages from {len(corpus)} total")
    
    print(f"Embedding {len(texts)} passages with {args.model}...")
    
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(args.model)
    
    embeddings = model.encode(
        texts,
        batch_size=args.batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=False,  # normalization done in experiment script
    )
    
    print(f"Embeddings shape: {embeddings.shape}")
    print(f"Saving to {args.output}...")
    np.save(args.output, embeddings.astype(np.float32))
    print("Done.")


if __name__ == "__main__":
    main()
