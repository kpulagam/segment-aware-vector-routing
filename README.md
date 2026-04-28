# HMRC: Heterogeneous Multi-Representative Centroids

[![Paper](https://img.shields.io/badge/Paper-PDF-red)](paper/HMRC_paper.pdf)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

**Improving Vector Database Routing for Heterogeneous Segments**

## 🎯 Problem

Vector databases use segment-based architectures where each segment is represented by a **single mean centroid** for query routing. When segments contain vectors from **multiple topics** (heterogeneous), the mean centroid fails to represent the segment, causing **routing failures**.

<p align="center">
<img src="paper/figures/fig1_problem_validation.png" width="600">
</p>

**Our finding:** Routing recall drops from **100%** (coherent) → **64.4%** (3 topics mixed) → **14.6%** (random)

## 💡 Solution

**HMRC** uses k-means clustering within each segment to generate **multiple representative centroids**, capturing the internal topic structure.

<p align="center">
<img src="paper/figures/fig2_hmrc_solution.png" width="600">
</p>

**Result:** +26% improvement on realistic heterogeneous segments with only 3× centroid storage overhead.

## 📊 Key Results

| Segmentation | Baseline | HMRC-3 | Improvement |
|--------------|----------|--------|-------------|
| Coherent     | 100.0%   | 100.0% | 0% (no harm)|
| Semi-3 (realistic) | 64.4% | **81.2%** | **+26%** |
| Random       | 14.6%    | 16.2%  | +11%        |

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/kpulagam/segment-aware-vector-routing.git
cd segment-aware-vector-routing
pip install -r requirements.txt
```

### Basic Usage

```python
from src.hmrc import HMRCIndex, MeanCentroidIndex
import numpy as np

# Your vectors and segment assignments
vectors = np.random.randn(10000, 384).astype(np.float32)
assignments = np.repeat(np.arange(20), 500)  # 20 segments

# Baseline
baseline = MeanCentroidIndex()
baseline.fit(vectors, assignments)

# HMRC with 3 representatives per segment
hmrc = HMRCIndex(n_representatives=3)
hmrc.fit(vectors, assignments)

# Route queries
queries = np.random.randn(100, 384).astype(np.float32)
baseline_routes = baseline.route(queries, k=10)
hmrc_routes = hmrc.route(queries, k=10)
```

### Run Experiments

```bash
# Generate embeddings (first time only)
python experiments/generate_embeddings.py --max-passages 100000

# Run full paper experiments
python experiments/run_paper_experiments.py \
    --embeddings data/embeddings/msmarco_100000.npy \
    --output results/
```

## 📁 Repository Structure

```
segment-aware-vector-routing/
├── src/
│   ├── hmrc.py              # Core HMRC implementation
│   ├── segmentation.py      # Segment creation strategies (Coherent/Semi-t/Random/Time-batched)
│   └── evaluation.py        # Routing recall evaluation
├── experiments/
│   ├── main.py              # Run paper experiments end-to-end
│   ├── exp10_routing_time.py
│   ├── paper_results_MSMARCO-100K_20260413_121345/
│   └── paper_results_NQ_20260413_123447/
├── paper/
│   └── HMRC_IEEE_Access.pdf
├── requirements.txt
├── LICENSE
└── README.md
```

## 🔬 How It Works

### Baseline (Mean Centroid)
```
Segment with 3 topics mixed:
┌─────────────────────────────────┐
│  ●●●    ▲▲▲    ■■■              │
│   ●●     ▲▲     ■■              │
│      ★ (mean - represents nothing well)
└─────────────────────────────────┘
```

### HMRC (Multiple Representatives)
```
Segment with 3 topics mixed:
┌─────────────────────────────────┐
│  ●●●    ▲▲▲    ■■■              │
│   ●●     ▲▲     ■■              │
│   ★1     ★2     ★3  (k-means centroids)
└─────────────────────────────────┘
```

## 📈 Experiments

### Experiment 1: Problem Validation
Shows routing recall degrades with segment heterogeneity.

### Experiment 2: HMRC Solution
Compares HMRC variants (k=2,3,5,7) against baseline.

### Experiment 3: Topics Ablation
Shows HMRC benefit increases with heterogeneity level.

### Experiment 4: Representatives Ablation
Shows k=3 is optimal (diminishing returns after).

### Experiment 5: Segment Sizes
Shows HMRC works across different segment sizes.

## 📝 Citation

```bibtex
@article{pulagam2026hmrc,
  title   = {HMRC: Heterogeneous Multi-Representative Centroids for Improved Vector Database Routing},
  author  = {Pulagam, Kishore},
  journal = {IEEE Access},
  year    = {2026},
  note    = {Under review}
}
```

## 📚 References

- [FAISS](https://github.com/facebookresearch/faiss) - Johnson et al., 2019
- [ScaNN](https://github.com/google-research/google-research/tree/master/scann) - Guo et al., 2020
- [HNSW](https://github.com/nmslib/hnswlib) - Malkov & Yashunin, 2018
- [Milvus](https://milvus.io/) - Wang et al., 2021

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.
