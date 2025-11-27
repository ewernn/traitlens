# traitlens v0.4

A minimal toolkit for extracting and analyzing trait vectors from transformer language models.

## What is traitlens?

traitlens provides low-level primitives for:
- **Hook management** - Register hooks on any module in your model
- **Activation capture** - Store and retrieve activations during forward passes
- **Trait computations** - Extract trait vectors and analyze their dynamics
- **Extraction methods** - Multiple algorithms for extracting trait vectors from activations

Think of it as "pandas for trait analysis" - we provide the building blocks, you design the analysis.

## Installation

```bash
# From the per-token-interp repo
cd traitlens
pip install -e .

# Or just import directly (no installation needed)
from traitlens import HookManager, ActivationCapture, mean_difference
```

## Quick Start

### Basic Activation Capture

```python
from traitlens import HookManager, ActivationCapture
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load your model (any HuggingFace model works)
model = AutoModelForCausalLM.from_pretrained("your-model")
tokenizer = AutoTokenizer.from_pretrained("your-model")

# Capture activations from any layer
capture = ActivationCapture()
with HookManager(model) as hooks:
    # Hook path depends on model architecture (e.g., "model.layers.N" for Llama/Gemma)
    hooks.add_forward_hook("model.layers.12", capture.make_hook("layer_12"))

    # Run forward pass
    inputs = tokenizer("Hello world", return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=10)

# Get captured activations
activations = capture.get("layer_12")  # [batch, seq_len, hidden_dim]
```

### Extract Trait Vectors

```python
from traitlens import MeanDifferenceMethod, ProbeMethod, get_method

# Collect activations from positive examples (e.g., refusal)
pos_acts = torch.randn(100, 2304)  # [n_examples, hidden_dim]

# Collect activations from negative examples (e.g., compliance)
neg_acts = torch.randn(100, 2304)

# Method 1: Mean difference (simple baseline)
method = MeanDifferenceMethod()
result = method.extract(pos_acts, neg_acts)
trait_vector = result['vector']

# Method 2: Linear probe (supervised boundary)
method = ProbeMethod()
result = method.extract(pos_acts, neg_acts)
trait_vector = result['vector']
train_acc = result['train_acc']

# Method 2b: Sparse probe (L1 regularization - shows which dims matter)
result = method.extract(pos_acts, neg_acts, penalty='l1')
sparse_vector = result['vector']  # Many near-zero components

# Method 3: Factory function
method = get_method('ica')  # or 'probe', 'gradient', 'mean_diff', 'pca_diff', 'random_baseline'
result = method.extract(pos_acts, neg_acts)
```

### Analyze Temporal Dynamics

```python
from traitlens import compute_derivative, compute_second_derivative

# Track activation trajectory over tokens
trajectory = torch.stack(per_token_activations)  # [seq_len, hidden_dim]

# Compute velocity (rate of change)
velocity = compute_derivative(trajectory)

# Compute acceleration (change in rate of change)
acceleration = compute_second_derivative(trajectory)

# Find commitment point (where acceleration drops)
commitment = (acceleration.norm(dim=-1) < threshold).nonzero()[0]
```

### Hook Multiple Locations

```python
# Compare trait across different model components
# (paths vary by architecture - these are for Llama/Gemma style)
locations = {
    'residual': 'model.layers.12',
    'attention': 'model.layers.12.self_attn.o_proj',
    'mlp': 'model.layers.12.mlp.down_proj',
}

capture = ActivationCapture()
with HookManager(model) as hooks:
    for name, path in locations.items():
        hooks.add_forward_hook(path, capture.make_hook(name))

    # Single forward pass captures all locations
    outputs = model(**inputs)

# Analyze each location
for name in locations:
    acts = capture.get(name)
    print(f"{name}: shape={acts.shape}")
```

## Core Components

### HookManager
Manages forward hooks on any PyTorch model. Automatically cleans up when used as context manager.

### ActivationCapture
Stores activations during forward passes. Use `make_hook()` to create hook functions.

### Extraction Methods
- `MeanDifferenceMethod` - Simple mean difference (baseline)
- `ICAMethod` - Independent component analysis (requires scikit-learn)
- `ProbeMethod` - Linear probe via logistic regression (supports L1/L2 via `penalty` param)
- `GradientMethod` - Gradient-based optimization
- `PCADiffMethod` - PCA on per-example differences (RepE-style)
- `RandomBaselineMethod` - Random unit vector (sanity check, should get ~50% accuracy)
- `get_method(name)` - Factory function for any method

### Compute Functions
- `mean_difference()` - Compute mean difference of tensors
- `compute_derivative()` - Calculate velocity of trait expression
- `compute_second_derivative()` - Calculate acceleration
- `projection()` - Project activations onto trait vectors
- `cosine_similarity()` - Compare vectors
- `normalize_vectors()` - Normalize to unit length
- `magnitude()` - Compute L2 norm of vectors
- `radial_velocity()` - Magnitude change between consecutive points
- `angular_velocity()` - Direction change between consecutive points
- `pca_reduce()` - Reduce activations to N dimensions via PCA
- `attention_entropy()` - Compute entropy of attention distribution

## Philosophy

traitlens is intentionally minimal. We provide:
- ✅ Easy hook management
- ✅ Simple activation storage
- ✅ Basic trait computations

We do NOT provide:
- ❌ Model wrappers
- ❌ Pre-defined hook locations
- ❌ Built-in analyses
- ❌ Visualization tools

Like numpy gives you arrays (not statistical tests), traitlens gives you activations (not interpretability methods). Build your own analysis on top.

## Requirements

- PyTorch
- transformers (for loading models, not core functionality)
- scikit-learn (optional, for ICA and Probe methods)

## License

Part of the per-token-interp project.