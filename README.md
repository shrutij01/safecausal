# Sparse Shift Autoencoders: Identifying Concepts from LLM Activations

A research framework for training and evaluating **Sparse Sparse Autoencoders (SSAEs)** on language model representations, with a focus on steering concepts, probing model behavior, and achieving identifiable representations at scale.

Latest version of the paper at [https://openreview.net/forum?id=dGQubVJQx6&referrer=%5Bthe%20profile%20of%20Shruti%20Joshi%5D(%2Fprofile%3Fid%3D~Shruti_Joshi1)](this link).

---

## Table of Contents

- [Motivation](#motivation)
  - [Why Constrained Optimization Instead of Sparsity Regularization?](#why-constrained-optimization-instead-of-sparsity-regularization)
  - [The Challenge of Identifiability at Scale](#the-challenge-of-identifiability-at-scale)
- [Key Features](#key-features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Architecture](#architecture)
  - [Core Components](#core-components)
  - [SSAE Training Pipeline](#ssae-training-pipeline)
  - [Evaluation Pipeline](#evaluation-pipeline)
- [Usage Guide](#usage-guide)
  - [Training SSAEs](#training-ssaes)
  - [Extracting Embeddings](#extracting-embeddings)
  - [Training Probes](#training-probes)
  - [K-Sweep Experiments](#k-sweep-experiments)
- [Data Formats](#data-formats)
- [Evaluation Metrics](#evaluation-metrics)
- [Project Structure](#project-structure)
- [Citation](#citation)

---

## Motivation

### Why Constrained Optimization Instead of Sparsity Regularization?

Traditional sparse autoencoders (SAEs) use **L1 regularization** (or variants like JumpReLU) to encourage sparsity:

```
L = MSE(x, x_hat) + λ * ||h||_1
```

While this approach is simple and widely used, it suffers from several fundamental problems:

#### 1. **The λ Tuning Problem**

With regularization, sparsity is a *side effect* rather than a guarantee. The hyperparameter λ must be carefully tuned:
- **Too small**: Dense, uninterpretable representations
- **Too large**: Excessive reconstruction loss, information destruction
- **Dataset-dependent**: Optimal λ varies with data distribution, making cross-dataset comparison unreliable

#### 2. **Non-Stationary Training Dynamics**

As training progresses, the effective sparsity level drifts unpredictably. Early in training, the model may be too sparse; later, it may become too dense. This makes it impossible to guarantee a consistent sparsity level across different runs or datasets.

#### 3. **Trade-off Opacity**

The regularization coefficient λ obscures the actual trade-off being made. Users cannot specify "I want exactly 10% of neurons active" — they can only adjust λ and observe what sparsity emerges.

#### Our Solution: Constrained Optimization via Cooper

We reformulate SAE training as a **constrained minimization problem**:

```
minimize    MSE(x, x_hat)
subject to  ||h||_1 / (batch_size * hidden_dim) ≤ target_sparsity
```

This formulation, implemented using the [Cooper library](https://github.com/cooper-org/cooper), provides:

| Benefit | Description |
|---------|-------------|
| **Exact Sparsity Control** | Specify the exact sparsity level you want (e.g., `target=0.1` for 10% active neurons) |
| **Automatic Trade-off Resolution** | The dual optimizer (Lagrangian multiplier) automatically finds the minimal reconstruction loss achievable at your target sparsity |
| **Stable Training** | Extrapolation-based optimization (ExtraAdam) eliminates oscillations common in primal-dual methods |
| **Reproducibility** | Same sparsity target yields comparable results across datasets and runs |
| **Principled Scheduling** | Linear warmup from dense to target sparsity ensures stable early training |

#### Implementation Details

Our SSAE class inherits from `cooper.ConstrainedMinimizationProblem`:

```python
class SSAE(cooper.ConstrainedMinimizationProblem):
    def compute_cmp_state(self, model, delta_z, loss_type="relative"):
        delta_z_hat, h = model(delta_z)
        loss = self._LOSS[loss_type](delta_z, delta_z_hat)

        # Sparsity as a hard constraint, not a penalty
        ineq = h.abs().sum() / (batch_size * hid) - self.level
        constraint_state = cooper.ConstraintState(violation=ineq)

        return cooper.CMPState(loss=loss, observed_constraints=...)
```

The optimizer uses **extrapolation** (looking ahead to simulate the other player's move) which dramatically reduces oscillations:

```python
coop_optimizer = cooper.optim.ExtrapolationConstrainedOptimizer(
    cmp=ssae,
    primal_optimizers=ExtraAdam(dict.parameters(), lr=5e-4),
    dual_optimizers=ExtraAdam(ssae.dual_parameters(), lr=2.5e-4, maximize=True),
)
```

---

### The Challenge of Identifiability at Scale

A central question in representation learning is: **can we recover the true underlying factors of variation?** This is the problem of *identifiability*.

#### The Fundamental Problem

Standard autoencoders are **non-identifiable**: for any learned representation `h`, there exist infinitely many rotations `Rh` that achieve identical reconstruction loss. This means:
- Features learned by SAEs may not correspond to meaningful concepts
- Different random seeds produce incomparable representations
- Interpretability claims become unfalsifiable

#### Why Identifiability is Hard at Scale

Several factors make identifiability particularly challenging in modern LLM settings:

| Challenge | Description |
|-----------|-------------|
| **Unknown Ground Truth** | We don't know how many "true concepts" exist in LLM representations |
| **Superposition** | LLMs encode more concepts than they have dimensions, violating standard identifiability assumptions |
| **Distributional Shift** | Training data distributions differ from deployment, breaking assumptions about source independence |
| **Scale** | With hidden dimensions of 65,536+, exhaustive verification is computationally infeasible |

#### Our Approach: Subtle but Principled Design Choices

We address identifiability through several interconnected design decisions:

##### 1. **Difference-Based Training (Δz = z₂ - z₁)**

Instead of training on raw activations, we train on *differences* between activation pairs:

```python
def __getitem__(self, idx):
    if len(self.data.shape) == 3:  # Paired data: (N, 2, rep_dim)
        return torch.from_numpy(self.data[idx, 1] - self.data[idx, 0])
    else:  # Sequential data: (N, rep_dim)
        return torch.from_numpy(self.data[idx + 1] - self.data[idx])
```

**Why this helps identifiability:**
- Removes shared/static information, isolating concept-specific variation
- Creates a natural "intervention" structure amenable to causal identification
- Reduces sensitivity to mean-shift distributional changes

##### 2. **Unit-Norm Decoder Columns**

We enforce unit L2 norm on decoder columns throughout training:

```python
@torch.jit.script
def renorm_decoder_cols_(W: Tensor, eps: float = 1e-8) -> None:
    col_norms = W.norm(p=2, dim=0, keepdim=True).clamp_min_(eps)
    W.data.div_(col_norms)

@torch.jit.script
def project_decoder_grads_(W: Tensor) -> None:
    """Project gradients to tangent space of unit-norm constraint."""
    g = W.grad
    if g is None:
        return
    col_dot = (W * g).sum(dim=0, keepdim=True)
    g.sub_(W * col_dot)  # g ← g − w(wᵀg)
```

**Why this helps identifiability:**
- Removes scale ambiguity (features can't be arbitrarily rescaled)
- Decoder columns become interpretable as "concept directions" in activation space
- Enables meaningful comparison of feature importance via coefficient magnitude

##### 3. **ZCA Whitening (Optional)**

Pre-processing with ZCA whitening decorrelates input features:

```python
class DataWhitener:
    def fit(self, X):
        cov = np.cov(X_centered, rowvar=False)
        eigvals, eigvecs = np.linalg.eigh(cov)
        D_inv_sqrt = np.diag(1.0 / np.sqrt(eigvals + epsilon))
        self.whitening_matrix = eigvecs @ D_inv_sqrt @ eigvecs.T  # ZCA
```

**Why this helps identifiability:**
- Under certain generative models (e.g., ICA), whitening + sparsity is sufficient for identification
- Equalizes variance across dimensions, preventing dominant features from dominating learning
- Improves optimization landscape by reducing condition number



## Key Features

- **Constrained Sparse Autoencoders**: Exact sparsity control via Lagrangian optimization
- **Difference-Based Training**: Learn from activation changes rather than raw activations
- **Multi-Concept Probing**: Evaluate feature interpretability across 13 binary concepts
- **K-Sweep Analysis**: Systematic evaluation of probe performance vs. feature count
- **Multiple Baselines**: Standard SAE (ReLU), JumpReLU, and PCA implementations
- **Experiment Tracking**: Full Weights & Biases integration

---

## Installation

```bash
# Clone the repository
git clone https://github.com/your-org/safecausal.git
cd safecausal

# Create environment (Python 3.10+)
conda create -n ssae python=3.10
conda activate ssae

# Install dependencies
pip install torch transformers datasets scikit-learn scipy
pip install cooper wandb h5py pyyaml tqdm python-box
```

## Quick Start

### 1. Extract Embeddings from a Language Model

```bash
python ssae/store_embeddings.py \
    --model "EleutherAI/pythia-70m" \
    --dataset labeled-sentences \
    --layer 5 \
    --output ./embeddings/pythia70m_layer5.h5
```

### 2. Train an SSAE

```bash
python ssae/ssae.py \
    ./embeddings/pythia70m_layer5.h5 \
    ./configs/pythia70m.yaml \
    --oc 1024 \
    --target 0.1 \
    --epochs 15000 \
    --lr 5e-4
```

## Architecture

### SSAE Training Pipeline

```
Input: Uniformly sampled sentences (s₁, s₂) with concept labels
                    │
                    ▼
┌─────────────────────────────────────────┐
│  Extract LM activations at layer L      │
│  z₁ = LM(s₁)[L], z₂ = LM(s₂)[L]        │
└─────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────┐
│  Compute differences: Δz = z₂ - z₁      │
│  (Isolates concept-specific variation)  │
└─────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────┐
│  Optional: ZCA Whitening                │
│  Δz_white = W_zca @ Δz                  │
└─────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────┐
│  SSAE Forward Pass:                     │
│  1. LayerNorm(Δz)                       │
│  2. h = Encoder(Δz - decoder_bias)      │
│  3. Normalize h                         │
│  4. Δz_hat = Decoder(h)                 │
└─────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────┐
│  Constrained Optimization:              │
│  min MSE(Δz, Δz_hat)                    │
│  s.t. ||h||₁ / (B × D) ≤ target         │
│                                          │
│  Cooper: Lagrangian + Extrapolation     │
└─────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────┐
│  Post-step: Project decoder to          │
│  unit-norm columns                       │
└─────────────────────────────────────────┘
```


## Usage Guide

### Training SSAEs

#### Basic Training

```bash
python ssae/ssae.py \
    /path/to/embeddings.h5 \
    /path/to/config.yaml \
    --oc 1024 \
    --target 0.1 \
    --epochs 15000
```

#### Configuration Options

| Argument | Default | Description |
|----------|---------|-------------|
| `--oc` | 10 | Output/hidden dimension (number of features) |
| `--target` | 0.1 | Target sparsity level (fraction of active neurons) |
| `--warmup` | 2000 | Epochs before sparsity constraint begins |
| `--schedule` | 5000 | Epochs to linearly decrease to target sparsity |
| `--lr` | 5e-4 | Primal learning rate |
| `--norm` | ln | Normalization type: ln (LayerNorm), gn, bn |
| `--loss` | relative | Loss type: relative (scale-invariant) or absolute |
| `--whiten` | False | Apply ZCA whitening to input |
| `--seed` | 0 | Random seed for reproducibility |

#### Config YAML Format

```yaml
# configs/pythia70m.yaml
model: "EleutherAI/pythia-70m"
rep_dim: 512  # Hidden dimension of the LM layer
layer: 5
```

### Extracting Embeddings

```bash
python ssae/store_embeddings.py \
    --model "google/gemma-2-2b-it" \
    --dataset labeled-sentences \
    --layer 12 \
    --batch-size 32 \
    --output ./embeddings/gemma2b_layer12.h5
```



## Data Formats

### Embedding HDF5 Format

```
embeddings.h5
├── cfc_train/     # Training embeddings, shape: (N, 2, rep_dim) or (N, rep_dim)
├── cfc_test/      # Test embeddings
└── metadata/      # Model info, layer, etc.
```

### Trained Model Directory

```
model_dir/
├── weights.pth    # Model state dict
├── cfg.yaml       # Training configuration
└── whitening_params.npz  # (if whitening enabled)
```
---

## Project Structure

```
safecausal/
├── ssae/                          # Core SSAE implementation
│   ├── ssae.py                   # SSAE training with Cooper
│   ├── store_embeddings.py       # LM embedding extraction
│   └── __init__.py
│
├── scripts/                       # Training and evaluation
│   ├── train_probes.py           # Main probe training script
│   ├── probe_data.py             # Pairwise dataset creation
│   ├── evaluate_*.py             # Various evaluation scripts
│   └── plots/                    # Visualization scripts
│
├── baselines/                     # Baseline implementations
│   ├── saes.py                   # Standard SAE, JumpReLU
│   ├── train_saes.py             # Baseline training
│   └── pca.py                    # PCA baseline
│
├── utils/                         # Utilities
│   ├── data_utils.py             # Data loading and processing
│   ├── metrics.py                # MCC, RDC, correlation metrics
│   └── debug_tools.py            # Debugging utilities
│
├── data/                          # Datasets
│   ├── labeled-sentences/        # Main evaluation dataset
│   ├── categorical_dataset/      # Synthetic data generation
│   └── *.json                    # Reference datasets
│
├── loaders/                       # Safe data loading
│   └── testdataloader.py
│
├── tests/                         # Test suite
│   └── test_safe_loader.py
│
├── configs/                       # Model configurations
│
├── K_SWEEP_USAGE.md              # K-sweep experiment guide
├── PAIRWISE_MULTICLASS_USAGE.md  # Pairwise probing guide
├── MCC_COMPUTATION_FIX.md        # Metric aggregation methodology
└── README.md                      # This file
```

---

## Supported Models

The framework has been tested with:

| Model | Hidden Dim | Recommended Layer |
|-------|------------|-------------------|
| EleutherAI/pythia-70m | 512 | 5 |
| EleutherAI/pythia-160m | 768 | 8 |
| EleutherAI/pythia-410m | 1024 | 12 |
| google/gemma-2-2b-it | 2304 | 12-18 |
| meta-llama/Llama-3.1-8b | 4096 | 16-24 |

---

## Experiment Tracking

All experiments are tracked via Weights & Biases:

```python
logger = Logger(entity=__, project=__, config=enhanced_config)

# Tracked metrics per epoch:
# - recon_loss: Reconstruction MSE
# - l0_sparsity: Actual sparsity (fraction of active neurons)
# - sparsity_target: Current scheduled target
# - constraint_violation: Distance from sparsity constraint
# - gpu_memory_*: Memory usage statistics
```

---

## Research Questions

This codebase is designed to investigate:

1. **Do SSAEs achieve better identifiability than regularized SAEs?**
   - Compare MCC scores across methods

2. **How sparse can we go while maintaining concept steering capability?**
   - K-sweep experiments reveal the sparsity-performance trade-off

3. **Do difference-based representations improve causal identifiability?**
   - Compare Δz training vs. raw z training

---

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Run tests (`python -m pytest tests/`)
4. Format code (`black . && isort .`)
5. Commit changes (`git commit -m 'Add amazing feature'`)
6. Push to branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

---

## License

[Add license information]

---

## Citation

If you use this codebase in your research, please cite:

```bibtex
@misc{joshi2025identifiablesteeringsparseautoencoding,
      title={Identifiable Steering via Sparse Autoencoding of Multi-Concept Shifts},
      author={Shruti Joshi and Andrea Dittadi and Sébastien Lachapelle and Dhanya Sridhar},
      year={2025},
      eprint={2502.12179},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2502.12179},
}
```

---

## Acknowledgments

- [Cooper](https://github.com/cooper-org/cooper) for constrained optimization
- [icebeem](https://github.com/ilkhem/icebeem) for MCC metric implementation
- EleutherAI for Pythia models
- Google for Gemma models

