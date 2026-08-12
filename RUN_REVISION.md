# IEEE Access Revision — Run Book

Everything below was implemented and smoke-tested. Experiments in **Part A
already ran** (results in `experiments/revision_results/`). **Part B needs
your machine** (internet + sentence-transformers; Apple GPU used
automatically via MPS).

## Part A — already done (re-run locally to get final sklearn numbers)

The sandbox runs used a NumPy k-means fallback; numbers differ from
scikit-learn by ~1–2 recall points. Re-run these on your machine (fast,
uses real sklearn) so the paper tables come from the canonical stack:

```bash
cd experiments
# kRt comparison + multi-seed CIs (~15 min total)
python exp12_krt_comparison.py --embeddings ../data/embeddings/msmarco_100000.npy \
    --dataset-name MSMARCO-100K --output revision_results/exp12 --workers 4
python exp12_krt_comparison.py --embeddings data/embeddings/nq_dataset.npy \
    --dataset-name NQ --output revision_results/exp12 --workers 4
python exp12_krt_comparison.py --embeddings ../data/embeddings/msmarco_100000.npy \
    --dataset-name MSMARCO-100K --output revision_results/exp12 --workers 4 \
    --strategies coherent timebatch
python exp12_krt_comparison.py --embeddings data/embeddings/nq_dataset.npy \
    --dataset-name NQ --output revision_results/exp12 --workers 4 \
    --strategies coherent timebatch

# dynamic workloads (~10 min)
python exp13_dynamic_workloads.py --embeddings ../data/embeddings/msmarco_100000.npy \
    --dataset-name MSMARCO-100K --output revision_results/exp13
python exp13_dynamic_workloads.py --embeddings data/embeddings/nq_dataset.npy \
    --dataset-name NQ --output revision_results/exp13
```

Then update Tables `tab:krtcompare` / `tab:dynamic` in the .tex if any
number moved by more than a point (LaTeX comments mark every number that
came from the sandbox run).

## Part B — needs internet + embedding model (order matters)

```bash
pip install sentence-transformers beir tqdm
cd experiments

# 1. Real-query bundles (NQ ~40 min embed on MPS; msmarco needs ~5 GB disk)
python prepare_revision_datasets.py --task nq
python prepare_revision_datasets.py --task msmarco

# 2. Multi-domain corpus: SciFact+NFCorpus+FiQA+ArguAna+SciDocs (~100K docs,
#    5 real domains, real queries; ~45 min)
python prepare_revision_datasets.py --task mdm

# 3. Second embedding family (BGE-base, 768d) for NQ + MDM (~2-3 h on MPS)
python prepare_revision_datasets.py --task second

# 4. 1M-vector corpus for the scale experiment (~2-4 h embed)
python prepare_revision_datasets.py --task scale
```

Then run the evaluations (all resumable / append-safe):

```bash
# Real queries vs held-out vectors (fills Section "Real Query Distributions")
python exp14_real_queries.py --bundle data/revision/nq100k_v2.npz \
    --dataset-name NQ-100K-v2
python exp14_real_queries.py --bundle data/revision/msmarco100k_v2.npz \
    --dataset-name MSMARCO-100K-v2
# Multi-domain: add the 'domain' strategy (real domains as topics)
python exp14_real_queries.py --bundle data/revision/mdm100k.npz \
    --dataset-name MDM-100K --strategies semi timebatch domain

# Second embedding family — rerun the core comparison on BGE embeddings
python exp14_real_queries.py --bundle data/revision/nq100k_v2_bge.npz \
    --dataset-name NQ-100K-v2-BGE
python exp14_real_queries.py --bundle data/revision/mdm100k_bge.npz \
    --dataset-name MDM-100K-BGE --strategies semi timebatch domain

# Scale to 1M
python exp15_scale.py --embeddings data/revision/msmarco_1m.npy \
    --output revision_results/exp15

# FAISS end-to-end on the new bundles (optional but strong):
python exp11_faiss_nprobe.py --embeddings data/revision/msmarco_1m.npy ...

# Figures for everything found under revision_results/
python analyze_revision.py --results revision_results
```

## What goes where in the paper

| Reviewer concern | Experiment | Paper location |
|---|---|---|
| Novelty vs kRt/Pyramid (R1, R2) | exp12 | Sec. "Comparison with Sub-Cluster Routing at Equal Budget" + Related Work II-E |
| k-means variance / CIs (R2) | exp12 seeds | "Statistical protocol" in Setup; all new tables |
| Updates/deletes/tenant growth (R1) | exp13 | Sec. "Dynamic Workloads" |
| Larger scale (R1) | exp15 | Sec. "Effect of Corpus Scale" (update with 1M numbers) |
| Real query distributions (R2) | exp14 | Sec. "Real Query Distributions" (fill XX placeholders) |
| More models/datasets (R1, R2) | exp14 on MDM/BGE bundles | extend Setup + new results table |
| FAISS implementation details (R2) | — | new "Implementation details" paragraph in Sec. V-D |
| Limitations (R1, R2) | exp12 random + exp13 tenants | Discussion "When HMRC does not help" + Conclusion |
