---

# High-Dimensional Cell Morphological Predictive Pipeline

An optimized, production-ready machine learning framework designed for analyzing high-dimensional, highly collinear cell morphological and phenotypic datasets (e.g., Cell Painting, high-content screening assays). 

This repository implements a robust, memory-safe pipeline that automatically handles extreme multicollinearity, conducts automated hyperparameter exploration using Bayesian optimization, and fits a high-performance gradient boosted tree architecture—all constrained to a stable, single-threaded execution layout to prevent system deadlocks.

---

## 🧬 Domain Context & Architecture

In high-content screening and cellular profiling, morphological features (e.g., cell area, texture, intensity, granularity) often exhibit intense correlation structures due to shared biological pathways. Feeding raw, redundant feature matrices directly into predictive models can result in erratic tree splits, inflation of feature importance, and severe overfitting.

This pipeline introduces a rigorous, three-stage separation between data preprocessing and model optimization:


```

[Raw Morphological Features]
│
▼
┌──────────────────────────────────────────┐
│  1. Run-Once Pre-computation Layer      │  <- Heavy matrix math executed ONCE
│     • Pairwise Correlation Distance      │
│     • Global Target Mutual Information   │
└────────────┬─────────────────────────────┘
│
▼
┌──────────────────────────────────────────┐
│  2. Optuna Optimization Loop             │  <- Explores parameter space sequentially
│     • Custom Cluster Feature Selector    │  <- Enforces 'distance' metric to block 1-feature traps
│     • LightGBM Direct Dataset Training   │  <- Strictly single-threaded (OMP/MKL locked)
└────────────┬─────────────────────────────┘
│
▼
┌──────────────────────────────────────────┐
│  3. Production Re-Fit & Validation       │  <- Trains champion model
│     • Independent Test Set Inference     │  <- Ultimate validation threshold (AUC > 0.95)
└──────────────────────────────────────────┘

```

### Core Components

1. **Run-Once Pre-computation Layer:** Calculates global Mutual Information (MI) scores and builds a hierarchical clustering dendrogram (Ward/Average linkage) outside the tuning loop. This isolates computationally expensive matrix operations, reducing downstream optimization trial runtime to just a few seconds.
2. **Precomputed Cluster Selector:** A custom scikit-learn-compatible transformer that slices the feature dendrogram based on absolute distance thresholds. It compresses collinear feature blocks by picking only the single top-performing feature per cluster based on MI, completely removing redundant biological noise.
3. **Core-Locked LightGBM Engine:** Uses LightGBM's native C-level Dataset API rather than scikit-learn wrapper abstractions. This prevents thread oversubscription and memory leakage across trials, maintaining deterministic performance.

---

## 🛠️ Installation & Environment Setup

This project uses explicit system environment locks to prevent low-level OpenMP/Intel MKL thread contention, which commonly causes python instances to freeze when running tree-based algorithms inside search loops.

### Prerequisites
* Python 3.10+
* Conda or virtualenv

### Setup Instructions

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/your-organization/cell-morphology-pipeline.git](https://github.com/your-organization/cell-morphology-pipeline.git)
   cd cell-morphology-pipeline

```

2. **Install core dependencies:**
```bash
pip install numpy pandas scipy scikit-learn lightgbm optuna

```


3. **Required Environment Variable Declaration:**
To guarantee C++ compiler-level stability, ensure the following single-thread locks are registered at the absolute top of your execution environment or main execution script:
```python
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

```




### Expected Output Logs

During execution, the script will output real-time optimization insights. Thanks to the absolute `distance` metric optimization, search trials converge efficiently on high-performance plateaus:

```text
[STEP 1/3] Running heavy calculations ONCE outside the loop...
Pre-computation phase complete.

[STEP 2/3] Initializing 100% Sequential Optuna Loop...
[I 2026-05-26 14:52:10] Trial 0 finished with value: 0.9782 and parameters: {...}
[I 2026-05-26 14:52:13] Trial 1 finished with value: 0.9641 and parameters: {...}

=== OPTUNA ANALYSIS HIGHLIGHTS ===
Peak Validation AUC: 0.9814
Optimal Feature Count: 34

[STEP 3/3] Building Final Champion Configuration...
Training final booster...
 -> Active convergence reached at tree #242

============================================================
               FINAL PRESENTATION READOUT
============================================================
Independent Holdout Test Dataset AUC: 0.9791
------------------------------------------------------------
Classification Summary Table (Evaluated on Holdout Test Split):
                precision    recall  f1-score   support

 Control_Group       0.97      0.98      0.97     10024
   Phenotype_A       0.98      0.97      0.98      9976

      accuracy                           0.98     20000
============================================================

```

---

## 📊 Optimization Log Interpretation & Best Practices

When analyzing trial logs or expanding the optimization space, use the following operational parameters:

### The "Distance vs. Inconsistent" Guideline

* **Always use `criterion='distance'**` for flat thresholding.
* Avoid `criterion='inconsistent'` when dealing with uniformly correlated morphological markers. The local inconsistency metric collapses the feature matrix into a single master cluster, producing a 1-feature model that bottlenecks performance to a baseline AUC floor (~0.75). Slicing by absolute distance creates multiple robust feature groups, opening the path to high-performance (>0.95 AUC) metrics.

### Tuning the Search Space Scale

The search boundaries in this repository are pre-optimized to enforce light, single-threaded computational loops:

* **`num_leaves`**: Kept within `15` to `31`. Morphological signal spaces are dense; deep tree fracturing (`>63` leaves) exponentially increases computation times without introducing statistically significant generalizable performance.
* **`feature_fraction`**: Maintained between `0.4` and `0.7`. This forces the booster to subsample different sets of representative cluster features for every split, acting as a regularizer against cellular batch effects.

--
