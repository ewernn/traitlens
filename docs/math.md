# Mathematical Foundations

The math behind traitlens, explained honestly.

## Extraction Methods

All methods take positive activations `pos_acts [n_pos, d]` and negative activations `neg_acts [n_neg, d]` and produce a trait vector `v [d]`.

### Mean Difference

```
v = mean(pos) - mean(neg)
```

Fast baseline. Finds the direction connecting class centroids. Ignores variance structure—if classes overlap significantly, this will still point at the centroid difference even if a better separating direction exists.

```python
from traitlens import MeanDifferenceMethod
result = MeanDifferenceMethod().extract(pos_acts, neg_acts)
```

### Linear Probe (L2)

```
v = argmax_w P(y|x;w)  where P = sigmoid(w·x + b)
```

Logistic regression weights. Finds the optimal linear decision boundary via maximum likelihood. Better than mean diff when classes have different variances or when the best separator isn't the centroid direction.

```python
from traitlens import ProbeMethod
result = ProbeMethod().extract(pos_acts, neg_acts)  # L2 default
```

### Sparse Probe (L1)

Same objective with L1 penalty:
```
v = argmax_w [P(y|x;w) - λ||w||₁]
```

Drives most weights to exactly zero. Use this when you want to know which dimensions carry the trait signal. High sparsity = more interpretable.

```python
result = ProbeMethod().extract(pos_acts, neg_acts, penalty='l1')
top_dims = result['vector'].abs().topk(10).indices  # Which dims matter
```

### Gradient Optimization

```
v = argmax_{||v||=1} [mean(pos)·v - mean(neg)·v]
```

Constrained to unit sphere. Iteratively optimizes separation via gradient descent. Mathematically equivalent to normalized mean diff for this objective, but the framework extends to custom objectives (e.g., margin maximization, variance penalties).

```python
from traitlens import GradientMethod
result = GradientMethod().extract(pos_acts, neg_acts, num_steps=100, lr=0.01)
```

### PCA on Differences (RepE-style)

```
diffs_i = pos_i - neg_i  (paired examples)
v = first principal component of diffs
```

Pairs positive and negative examples, computes per-pair differences, returns first PC. Unlike mean diff (which just computes centroids), this captures the variance structure of differences. From the Representation Engineering paper.

```python
from traitlens import PCADiffMethod
result = PCADiffMethod().extract(pos_acts, neg_acts)
```

### ICA (Independent Components)

```
X = AS  where S has independent components
v = column of mixing matrix A
```

Finds statistically independent directions in the combined data. Useful when multiple traits co-occur and you want to disentangle them. The `component_idx` parameter selects which independent component to use.

```python
from traitlens import ICAMethod
result = ICAMethod().extract(pos_acts, neg_acts, n_components=50, component_idx=0)
# result['separation_scores'] shows which component best separates classes
```

### Random Baseline

```
v ~ N(0, I), then v = v/||v||
```

Random unit vector. Should achieve ~50% accuracy. If it scores higher, your evaluation pipeline is broken.

```python
from traitlens import get_method
result = get_method('random_baseline').extract(pos_acts, neg_acts)
```

## Method Selection

| Method | When to Use | Strengths | Weaknesses |
|--------|-------------|-----------|------------|
| `mean_diff` | Quick baseline, well-separated classes | Fast, simple, no hyperparameters | Ignores variance, suboptimal boundary |
| `probe` (L2) | Standard choice, overlapping classes | Optimal linear separator | Requires sklearn, all dims contribute |
| `probe` (L1) | Need interpretability | Sparse, shows which dims matter | Slower, requires tuning C |
| `gradient` | Custom objectives, unit-norm required | Flexible, normalized output | Slower, local optima |
| `pca_diff` | Paired examples, RepE replication | Captures variance structure | Requires paired data |
| `ica` | Disentangling co-occurring traits | Finds independent factors | Requires sklearn, many hyperparameters |
| `random_baseline` | Sanity checking | Catches evaluation bugs | Not a real method |

**Default recommendation**: Start with `probe` (L2). Fall back to `mean_diff` if you need speed or don't have sklearn.

## Compute Operations

### Temporal Derivatives

For activation trajectory `x_t` over tokens:

```
velocity:     v_t = (x_{t+1} - x_t) / dt
acceleration: a_t = (v_{t+1} - v_t) / dt
```

Velocity measures how fast the representation changes. Acceleration drop indicates "commitment points" where the model has decided.

```python
from traitlens import compute_derivative, compute_second_derivative

trajectory = capture.get("layer_16")  # [seq_len, hidden_dim]
velocity = compute_derivative(trajectory)
acceleration = compute_second_derivative(trajectory)

# Find commitment point
commitment = (acceleration.norm(dim=-1) < threshold).nonzero()[0]
```

### Projections

```
score = (x · v) / ||v||  # with normalize_vector=True (default)
score = x · v            # with normalize_vector=False
```

Projects activations onto trait vector. Score indicates trait expression strength.

```python
from traitlens import projection
scores = projection(activations, trait_vector)  # [batch, seq_len]
```

### Radial and Angular Velocity

Decompose trajectory change into magnitude and direction:

```
radial:  Δ||x|| = ||x_{t+1}|| - ||x_t||     # magnitude change
angular: 1 - cos(x_t, x_{t+1})               # direction change (0=same, 2=opposite)
```

```python
from traitlens import radial_velocity, angular_velocity
rad_vel = radial_velocity(trajectory)  # Growing or shrinking?
ang_vel = angular_velocity(trajectory)  # Changing direction?
```

### Cosine Similarity

```
cos(a, b) = (a · b) / (||a|| ||b||)
```

Use this instead of dot product when comparing vectors. See "Normalization" section below for why.

```python
from traitlens import cosine_similarity
sim = cosine_similarity(vec1, vec2)  # -1 to 1
```

### PCA Reduction

Standard PCA: center data, compute covariance, take top eigenvectors.

```python
from traitlens import pca_reduce
reduced = pca_reduce(activations, n_components=2)  # For visualization
```

### Attention Entropy

```
H(p) = -Σ p_i log(p_i)
```

High entropy = diffuse attention (looking everywhere). Low entropy = focused attention (looking at few tokens).

```python
from traitlens import attention_entropy
entropy = attention_entropy(attention_weights)  # Per head, per query position
```

## Evaluation Metrics

### Accuracy

```
accuracy = (TP + TN) / (P + N)
threshold = (mean(pos_proj) + mean(neg_proj)) / 2  # default
```

Fraction correctly classified. Uses midpoint of means as default threshold.

### Effect Size (Cohen's d)

```
d = |mean(pos) - mean(neg)| / pooled_std
pooled_std = sqrt((std(pos)² + std(neg)²) / 2)
```

Separation in units of standard deviation. Guidelines: 0.2 = small, 0.5 = medium, 0.8 = large. Real trait vectors typically show d > 2.

### What "Good" Looks Like

| Metric | Poor | Okay | Good | Excellent |
|--------|------|------|------|-----------|
| Accuracy | < 60% | 60-75% | 75-90% | > 90% |
| Effect size | < 0.5 | 0.5-1.0 | 1.0-2.0 | > 2.0 |
| p-value | > 0.05 | < 0.05 | < 0.01 | < 0.001 |

**Important**: High accuracy ≠ valid trait. See "Caveats" section.

### Stability Metrics

- **Bootstrap stability**: Resample data, measure variance in separation. CV < 0.1 is stable.
- **Noise robustness**: Add noise to vector, measure separation drop. > 0.9 is robust.
- **Subsample stability**: Use half the data, compare separation. > 0.9 generalizes well.

```python
from traitlens.metrics import evaluate_vector
metrics = evaluate_vector(pos_acts, neg_acts, vector, include_stability=True)
```

### Vector Properties

- **Sparsity**: Fraction of near-zero components. Higher = more interpretable.
- **Effective rank**: Dimensions needed for 90% of mass. Lower = more concentrated.
- **Top-k concentration**: Mass in top 5% of dims. Higher = more interpretable.

## High-Dimensional Geometry

Three counterintuitive facts about high-dimensional spaces that affect trait analysis:

### 1. Random Vectors Are Nearly Orthogonal

In d dimensions, two random unit vectors have:
```
E[cos(a,b)] = 0
Var[cos(a,b)] ≈ 1/d
```

For d = 2304 (Gemma-2B), the standard deviation is ~0.02. This means:
- ~50% accuracy on unrelated traits is *expected*, not evidence of independence
- Cosine > 0.1 between trait vectors suggests real correlation

### 2. Volume Concentrates Near Surface

Most of the volume of a high-dimensional sphere is concentrated in a thin shell near the surface. Activations tend to live on a manifold, not fill the space uniformly.

### 3. Distances Concentrate

All points are approximately equidistant from each other. Euclidean distance becomes less meaningful; use cosine similarity instead.

## Normalization

**Always use cosine similarity, not dot product.**

Why: Activation magnitudes vary wildly across layers (often 10-15× difference). A dot product conflates magnitude with direction. Cosine similarity isolates the direction, which is what trait vectors capture.

```python
# Wrong: dot product
score = activations @ vector

# Right: cosine similarity (what evaluate_vector does by default)
score = (activations / activations.norm()) @ (vector / vector.norm())
```

The `evaluate_vector` function normalizes by default. Set `normalize=False` only if you specifically want magnitude effects.

## Honest Caveats

### Correlation ≠ Causation

These methods find directions that *correlate* with traits. They don't prove the direction *causes* the trait. A vector might:
- Capture a confound (e.g., "refusal" vector actually captures "formal tone")
- Reflect the trait's downstream effects rather than its cause
- Work via a mechanism unrelated to the trait itself

**Validation required**: Intervene on the vector (add/subtract from activations) and check behavioral effects.

### High Accuracy ≠ Real Trait

You can get 95% accuracy on a "trait" that isn't real:
- Spurious correlations in your dataset
- Leakage from prompt structure
- Model-specific artifacts

**Mitigation**: Test on held-out prompts, check cross-model generalization, validate with interventions.

### The Linear Representation Hypothesis is Contested

traitlens assumes traits are encoded as linear directions. This is convenient but not proven:
- Some traits may require nonlinear combinations
- Representation may be distributed across dimensions in non-additive ways
- What looks like one trait may be multiple entangled features

**Pragmatic stance**: Linear methods work surprisingly well empirically. Use them, but don't overinterpret.

### N is Usually Too Small

Most trait extraction uses 50-200 examples per class. With d = 2304 dimensions, this is heavily underdetermined. The methods regularize (probe) or reduce dimensionality (PCA, ICA), but:
- Vectors may not generalize to new prompts
- Bootstrap stability gives a lower bound on reliability
- More data is always better

**Minimum recommended**: 100+ examples per class, with diverse prompt structures.
