# Continuous-Time PairRE-based Temporal Knowledge Graph Embedding

## 1. Overview

This project aims to develop a **Temporal Knowledge Graph Embedding (TKGE)** model that learns continuous representations of entities and relations over time.

Unlike traditional temporal embedding models such as **TComplEx** or **TNTComplEx**, which rely on **discrete timestamp embeddings**, the proposed approach focuses on:

- **Continuous-time modeling**:  
  Time is represented as a real-valued variable, enabling both **interpolation** and **extrapolation** to unseen timestamps.

- **Hybrid temporal knowledge handling**:  
  The model jointly represents:

  - **Static facts** (time-invariant knowledge)
  - **Dynamic facts** (time-dependent knowledge)  
    via a **learnable gating mechanism**.

- **Extensibility toward spatial hierarchy modeling** (future work):  
  The framework is designed to support hierarchical structures such as geographic containment trees.

The model is evaluated on standard TKBC benchmarks including **ICEWS14**, **ICEWS05-15**, and **YAGO15k**.

---

## 2. Model Architecture

### 2.1 Scoring Function

The scoring function is inspired by **PairRE**, extended with continuous-time modulation and relation-wise gating:

\[
\varphi(h, r, t, m)
=
\left|
\left(
h \circ r^H - t \circ r^T
\right)
\circ
\left(
\alpha \cdot m + (1 - \alpha) \cdot \vec{1}
\right)
\right|
\]

Where:

- \( h, t \in \mathbb{R}^d \): embeddings of head and tail entities
- \( r^H, r^T \in \mathbb{R}^d \): relation-specific projection vectors (as in PairRE)
- \( m \in \mathbb{R}^d \): continuous-time embedding
- \( \alpha \in [0,1] \): relation-specific temporal gating parameter
- \( \vec{1} \): all-one vector preserving static information
- \( \circ \): Hadamard (element-wise) product

---

### 2.2 Continuous Time Embedding

Time is embedded using a **learnable cosine-based regression function**:

\[
m = \cos(W \cdot \tau + b)
\]

Where:

- \( \tau \in \mathbb{R} \): normalized continuous timestamp
- \( W \in \mathbb{R}^{d} \), \( b \in \mathbb{R}^{d} \): learnable parameters
- Output \( m \in \mathbb{R}^{d} \) matches the entity embedding dimension

---

### 2.3 Temporal Gating Mechanism

Each relation learns a gating coefficient:

\[
\alpha_r = \sigma(\text{Embedding}(r))
\]

- \( \alpha_r \approx 0 \): static relations
- \( \alpha_r \approx 1 \): dynamic relations

---

## 3. Implementation Guide

### Step 1: Time Preprocessing

1. Load `ts_id` from the TKBC dataset (e.g., ICEWS14).
2. Map timestamp IDs to real-valued time points using Min-Max scaling:

\[
\tau = \frac{t*{real} - t*{min}}{t*{max} - t*{min}} \times 100
\]

3. Construct a lookup table:

```
timestamp_id → normalized_time (float)
```

---

### Step 2: Model Implementation (`models.py`)

```python
class ContinuousTimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.W = nn.Parameter(torch.randn(dim))
        self.b = nn.Parameter(torch.zeros(dim))

    def forward(self, t):
        return torch.cos(t.unsqueeze(-1) * self.W + self.b)
```

Relation-wise gating:

```python
self.alpha = nn.Embedding(num_relations, 1)
alpha = torch.sigmoid(self.alpha(rel_ids))
```

Score computation:

```python
interaction = h * r_h - t * r_t
gate = alpha * m + (1 - alpha)
score = torch.abs(interaction * gate).sum(dim=-1)
```

---

### Step 3: Dataset & Training Pipeline

- Modify `TemporalDataset` to return normalized time values.
- Ensure optimizer updates entity embeddings, relation projections, time parameters \(W, b\), and gating \(\alpha\).

---

## 4. Evaluation Protocol

### 4.1 Baselines

- TComplEx
- TNTComplEx

Metrics:

- MRR
- Hits@1, Hits@3, Hits@10

Datasets:

- ICEWS14
- ICEWS05-15
- YAGO15k

---

### 4.2 Temporal Generalization

Evaluate on timestamps not seen during training to assess interpolation and extrapolation.

---

### 4.3 Gating Analysis

Analyze learned \( \alpha_r \) values:

- Static relations → \( \alpha_r \approx 0 \)
- Dynamic relations → \( \alpha_r \approx 1 \)

---

## 5. Project Structure

```
.
├── data/
│   ├── ICEWS14/
│   │   ├── train.pickle
│   │   ├── valid.pickle
│   │   ├── test.pickle
│   │   └── ts_id
├── models.py
├── datasets.py
├── learner.py
└── README.md
```

---

## 6. Summary

This project proposes a **continuous-time, relation-gated extension of PairRE** for Temporal Knowledge Graph Embedding, enabling smooth temporal modeling, unified handling of static and dynamic facts, and strong generalization to unseen timestamps.
