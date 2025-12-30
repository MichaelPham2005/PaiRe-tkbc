# Formula Verification Report

## Date: December 30, 2025

---

## ✅ All Mathematical Formulas Verified Correct

### 1. Relation-wise Gating Parameter

**Formula from specification:**

```
a ∈ ℝ^|R|
```

where each element `aᵣ` corresponds to one relation r.

**Implementation:** ✅ CORRECT

```python
self.alpha = nn.Embedding(sizes[1], 1)  # sizes[1] = |R|
nn.init.constant_(self.alpha.weight, 0.0)  # Initialize to 0
```

- Shape: (num_relations, 1) ✓
- Initialization: a = 0 → σ(0) = 0.5 (neutral baseline) ✓

---

### 2. Gating Function

**Formula from specification:**

```
G(m, r) = σ(aᵣ) · m + (1 - σ(aᵣ)) · 1
```

**Implementation:** ✅ CORRECT

```python
alpha = torch.sigmoid(self.alpha(x[:, 1].long()))  # σ(aᵣ)
gate = alpha * m + (1 - alpha) * torch.ones_like(m)
```

**Verification:**

- σ is sigmoid function ✓
- m is time embedding vector ✓
- 1 is ones vector (not scalar) ✓
- When α → 0: G → 1 (static, ignores time) ✓
- When α → 1: G → m (dynamic, full temporal) ✓

---

### 3. Continuous Time Embedding

**Formula from specification:**

```
m = cos(W · t + b)
```

where:

- t ∈ [-1, 1] (normalized continuous time)
- W, b ∈ ℝ^d (learnable parameters)
- m ∈ ℝ^d (time embedding)

**Implementation:** ✅ CORRECT

```python
class ContinuousTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.W = nn.Parameter(torch.randn(dim) * 0.01)
        self.b = nn.Parameter(torch.zeros(dim))

    def forward(self, t: torch.Tensor):
        return torch.cos(t.unsqueeze(-1) * self.W + self.b)
```

**Verification:**

- Element-wise cosine ✓
- Broadcasting correct ✓
- W initialized small (0.01) ✓
- b initialized zero ✓

---

### 4. Scoring Function

**Formula from specification:**

```
φ(h, r, t, m) = |(h ∘ r^H - t ∘ r^T) ∘ G(m, r)|
```

where:

- ∘ denotes Hadamard product (element-wise multiplication)
- | · | denotes L1 norm (sum of absolute values)
- Negative sign for distance → similarity conversion

**Implementation:** ✅ CORRECT

```python
def score(self, x: torch.Tensor):
    h = self.entity_embeddings(x[:, 0].long())
    r_h = self.relation_head(x[:, 1].long())
    t = self.entity_embeddings(x[:, 2].long())
    r_t = self.relation_tail(x[:, 1].long())

    time_continuous = x[:, 3].float()
    m = self.time_encoder(time_continuous)
    alpha = torch.sigmoid(self.alpha(x[:, 1].long()))
    gate = alpha * m + (1 - alpha) * torch.ones_like(m)

    interaction = (h * r_h - t * r_t) * gate  # Hadamard products
    score = -torch.norm(interaction, p=1, dim=-1)  # Negative L1 norm
    return score
```

**Verification:**

- h ∘ r^H: element-wise multiplication ✓
- t ∘ r^T: element-wise multiplication ✓
- Result ∘ G(m,r): element-wise multiplication ✓
- L1 norm: torch.norm(p=1) ≡ sum(abs()) ✓
- Negative sign: converts distance to similarity ✓

---

### 5. Forward Pass Consistency

**Implementation:** ✅ CORRECT (Fixed)

**Before fix:**

```python
score = -torch.abs(interaction).sum(dim=1)  # Manual L1
```

**After fix:**

```python
score = -torch.norm(interaction, p=1, dim=1)  # Consistent with score()
```

Both implementations are mathematically equivalent:

- `torch.norm(p=1)` = `torch.abs().sum()`
- But using `torch.norm(p=1)` is cleaner and consistent

---

## 📊 Verification Test Results

### Test 1: Alpha Shape

- Expected: (num_relations, 1)
- Actual: (10, 1)
- **Status: ✅ PASS**

### Test 2: Alpha Initialization

- Expected: ~0.0 (so σ(0) = 0.5)
- Actual: 0.000000
- **Status: ✅ PASS**

### Test 3: Time Embedding m = cos(W·t + b)

- Max difference: 0.0000000000
- **Status: ✅ PASS**

### Test 4: Gating Formula G(m,r)

- Formula match: 0.0000000000
- When α≈0, G≈1: 0.000000
- **Status: ✅ PASS**

### Test 5: Scoring Formula φ(h,r,t,m)

- Manual vs Model: 0.0000000000
- **Status: ✅ PASS**

### Test 6: L1 Norm Implementation

- Manual abs.sum: 10.000000
- torch.norm(p=1): 10.000000
- **Status: ✅ PASS**

### Test 7: Negative Sign Convention

- Reasoning verified: distance → similarity
- **Status: ✅ CORRECT**

---

## 🎯 Summary

**All mathematical formulas from the specification are correctly implemented:**

1. ✅ Relation-wise parameter vector a ∈ ℝ^|R|
2. ✅ Gating function G(m,r) = σ(aᵣ)·m + (1-σ(aᵣ))·1
3. ✅ Time embedding m = cos(W·t + b)
4. ✅ Scoring function φ = |(h∘r^H - t∘r^T)∘G(m,r)|
5. ✅ L1 norm for distance computation
6. ✅ Negative sign for similarity conversion
7. ✅ Hadamard products (element-wise multiplication)

**Minor fix applied:**

- Changed `torch.abs().sum()` to `torch.norm(p=1)` in forward() for consistency

**No other changes needed - implementation is mathematically correct!**

---

## 🚀 Ready for Training

The model implementation perfectly matches the mathematical specification.
You can proceed with training using:

```powershell
cd scripts
.\repreprocess_time.ps1  # Re-normalize to [-1, 1] if not done
.\train_continuous_pairre.ps1
```
