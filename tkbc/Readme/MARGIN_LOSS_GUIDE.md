# Margin-Based Loss Implementation

## Tổng quan

Implementation này thêm hàm loss theo đúng paper vào mô hình ContinuousPairRE, thay thế CrossEntropyLoss mặc định.

## Công thức Loss trong Paper

$$L = -\log \sigma(\gamma - f_r(h,t)) - \sum_{i=1}^{n} p(h_i', r, t_i') \log \sigma(f_r(h_i', t_i') - \gamma)$$

Trong đó:
- $f_r(h,t)$ là score function (giữ nguyên theo implementation của bạn)
- $\gamma$ là margin parameter (default: 9.0)
- $\sigma$ là sigmoid function
- $p(h_i', r, t_i')$ là self-adversarial weight cho negative samples:
  $$p(h_i', r, t_i') = \frac{\exp(\alpha \cdot f_r(h_i', t_i'))}{\sum_j \exp(\alpha \cdot f_r(h_j', t_j'))}$$
- $\alpha$ là adversarial temperature parameter

## Thay đổi Code

### 1. File `optimizers.py`

Thêm class `MarginBasedContinuousTimeOptimizer` với:

**Loss Components:**
- **Positive loss**: $-\log(\sigma(\gamma - f_r(h,t)))$ 
  - Maximize margin giữa γ và positive score
  - Score càng cao → loss càng thấp
  
- **Negative loss**: $-\sum p(h',r,t') \log(\sigma(f_r(h',t') - \gamma))$
  - Minimize negative scores so với margin
  - Negative score càng cao → loss càng cao

**Self-Adversarial Sampling:**
- Negative samples với score cao được weight nhiều hơn
- Giúp model focus vào "hard negatives"

### 2. File `learner.py`

Thêm arguments mới:
- `--use_margin_loss`: Enable margin-based loss
- `--gamma`: Margin parameter (default: 9.0)
- `--num_neg`: Số negative samples per positive (default: 64)
- `--adversarial_temp`: Temperature cho self-adversarial weighting (default: 1.0)

### 3. Training Scripts

- `train_continuous_pairre_margin.ps1` (Windows)
- `train_continuous_pairre_margin.sh` (Linux/Mac)

## Cách sử dụng

### Option 1: Sử dụng scripts có sẵn

**Windows:**
```powershell
cd tkbc/scripts
.\train_continuous_pairre_margin.ps1
```

**Linux/Mac:**
```bash
cd tkbc/scripts
chmod +x train_continuous_pairre_margin.sh
./train_continuous_pairre_margin.sh
```

### Option 2: Manual command

```bash
python learner.py \
    --dataset ICEWS14 \
    --model ContinuousPairRE \
    --rank 100 \
    --batch_size 1000 \
    --learning_rate 0.1 \
    --max_epochs 50 \
    --valid_freq 5 \
    --use_margin_loss \
    --gamma 9.0 \
    --num_neg 64 \
    --adversarial_temp 1.0
```

### So sánh với CrossEntropy

**Không dùng margin loss (mặc định - CrossEntropy):**
```bash
python learner.py --dataset ICEWS14 --model ContinuousPairRE --rank 100
```

**Dùng margin loss (theo paper):**
```bash
python learner.py --dataset ICEWS14 --model ContinuousPairRE --rank 100 --use_margin_loss
```

## Hyperparameters

### Gamma (Margin)
- **Default:** 9.0
- **Range:** 6.0 - 12.0
- Margin càng lớn → yêu cầu separation giữa positive/negative càng cao
- Nếu loss không giảm, thử giảm gamma

### Num Negative Samples
- **Default:** 64
- **Range:** 16 - 256
- Nhiều negatives → training chậm hơn nhưng có thể accurate hơn
- Ít negatives → training nhanh hơn

### Adversarial Temperature
- **Default:** 1.0
- **Range:** 0.0 - 2.0
- **0.0**: Uniform weighting (tất cả negatives có weight bằng nhau)
- **> 0**: Self-adversarial (hard negatives có weight cao hơn)
- Temperature càng cao → focus vào hard negatives càng mạnh

## Monitoring Training

Output hiển thị các metrics:
```
loss: 12.34   # Total loss
pos: 3.21     # Positive loss component
neg: 9.13     # Negative loss component
reg: 0.05     # Embedding regularization
cont: 0.01    # Time regularization
smooth: 0.0001 # Smoothness regularization
```

## So sánh Implementation

| Feature | CrossEntropy (Default) | Margin-Based (Paper) |
|---------|----------------------|---------------------|
| Loss function | Softmax + NLL | Sigmoid + Margin |
| Negative sampling | All entities | Sampled negatives |
| Weighting | Equal | Self-adversarial |
| Memory | Higher (scores all entities) | Lower (only sampled) |
| Speed | Slower | Faster |
| Theory | Classification | Ranking |

## Lưu ý

1. **Memory Usage:** Margin-based loss tiêu tốn ít memory hơn vì không cần tính score cho tất cả entities
2. **Speed:** Có thể nhanh hơn với batch size lớn
3. **Convergence:** Có thể cần tune gamma và adversarial_temp để có kết quả tốt nhất
4. **Compatibility:** Chỉ hoạt động với ContinuousPairRE model

## Troubleshooting

**Loss không giảm:**
- Giảm gamma (thử 6.0 hoặc 7.0)
- Tăng learning rate
- Giảm adversarial_temp (thử 0.5 hoặc 0.0)

**Training quá chậm:**
- Giảm num_neg (thử 32 hoặc 16)
- Tăng batch_size nếu GPU memory cho phép

**NaN loss:**
- Kiểm tra learning rate (có thể quá cao)
- Đảm bảo score function không trả về inf/nan
