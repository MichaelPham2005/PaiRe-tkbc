
# Relation-wise Temporal Gating for Continuous-Time TKGE

## 1. Motivation

In temporal knowledge graph embedding (TKGE), not all relations depend on time in the same way.

- **Static relations** (e.g., `birthPlace`, `capitalOf`) are invariant across time.
- **Dynamic relations** (e.g., `meet`, `attack`, `holdOffice`) strongly depend on temporal context.

Applying the same temporal modulation to all relations introduces unnecessary noise and degrades performance.

This project introduces a **relation-wise temporal gating mechanism** combined with **continuous-time modeling**, enabling the model to:
- Preserve static knowledge
- Emphasize temporal dynamics only where needed
- Improve MRR and temporal generalization

---

## 2. Mathematical Formulation

### 2.1 Relation-wise Gating Parameters

Instead of a single global gating scalar, we define a **relation-specific parameter vector**:

a ∈ R^{|R|}

Each element a_r corresponds to a relation r.
The effective temporal gate is obtained via a sigmoid:

alpha_r = sigmoid(a_r)

---

### 2.2 Continuous Time Embedding

Each timestamp is mapped onto a **continuous axis normalized to [-1, 1]**.

Time embedding is computed as:

m = cos(W * tau + b)

Where:
- tau ∈ [-1, 1]
- W, b ∈ R^d are learnable parameters
- m ∈ R^d matches entity embedding dimension

---

### 2.3 Gating Function

The **relation-aware temporal gating function** is defined as:

G(m, r) = alpha_r * m + (1 - alpha_r) * 1

Static relations (alpha_r ≈ 0) ignore time,
Dynamic relations (alpha_r ≈ 1) fully depend on time.

---

### 2.4 Final Scoring Function

The PairRE-based temporal scoring function becomes:

phi(h, r, t, m)
= | (h ∘ r^H - t ∘ r^T) ∘ G(m, r) |

---

## 3. Implementation Guide (PyTorch)

### 3.1 Relation-wise Gating Parameters

```python
self.alpha_embeddings = nn.Embedding(num_relations, 1)
nn.init.constant_(self.alpha_embeddings.weight, 0.0)  # sigmoid(0) = 0.5
```

---

### 3.2 Forward Pass

```python
alpha_r = torch.sigmoid(self.alpha_embeddings(r_idx))
m = torch.cos(self.W * cont.unsqueeze(-1) + self.b)
gating = alpha_r * m + (1 - alpha_r)
score = torch.norm((h * r_h - t * r_t) * gating, p=1, dim=-1)
```

---

## 4. Training Strategies

- Informed initialization toward dynamic relations
- Regularization to push alpha_r toward 0 or 1
- Monitoring relation polarization during training

---

## 5. Why This Improves MRR

- Protects static knowledge
- Focuses temporal modeling on dynamic relations
- Improves interpolation and extrapolation

---

## 6. Summary

Relation-wise temporal gating with continuous time normalization to [-1,1]
provides a principled and effective extension of PairRE for TKGE.
