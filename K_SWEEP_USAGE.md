# K-Sweep Experiment Usage Guide

## Overview

The k-sweep experiment evaluates probe performance as a function of the number of top features selected using **LR coefficient-based feature selection** (NOT gradient attribution).

**For difference-based SAEs**: Uses paired sentences with delta activations (acts_2 - acts_1) for proper change detection.

---

## Quick Start - Ready to Run Command

### For Pythia Models:
```bash
python scripts/train_probes.py \
    /network/scratch/j/joshi.shruti/ssae/labeled-sentences/paper_models/labeled-sentences_EleutherAI-pythia-70m-deduped_seed0 \
    --sweep-k-attribution \
    --k-values 1 2 5 10 25 50 100 \
    --batch-size 32 \
    --seed 42 \
    --metrics mcc
```

### For Gemma Models:
```bash
python scripts/train_probes.py \
    /network/scratch/j/joshi.shruti/ssae/labeled-sentences/paper_models/labeled-sentences_gemma-2-2b-it_seed0 \
    --sweep-k-attribution \
    --k-values 1 2 5 10 25 50 100 \
    --batch-size 28 \
    --seed 42 \
    --metrics mcc
```

### With Plot Output:
```bash
python scripts/train_probes.py \
    /network/scratch/j/joshi.shruti/ssae/labeled-sentences/paper_models/labeled-sentences_EleutherAI-pythia-70m-deduped_seed0 \
    --sweep-k-attribution \
    --k-values 1 2 5 10 25 50 100 \
    --batch-size 32 \
    --seed 42 \
    --metrics mcc \
    --plot-output results/k_sweep_pythia.png
```

### With Limited Samples (for faster testing):
```bash
python scripts/train_probes.py \
    /network/scratch/j/joshi.shruti/ssae/labeled-sentences/paper_models/labeled-sentences_gemma-2-2b-it_seed0 \
    --sweep-k-attribution \
    --k-values 1 5 10 50 \
    --batch-size 28 \
    --seed 42 \
    --metrics mcc \
    --max-samples 5000
```

---

## Feature Selection Method

✅ **Current Implementation**: `method="lr"` from `heuristic_feature_ranking_binary`
- Ranks features by `|probe.coef_[0]|` (absolute value of LR coefficients)
- Fast and efficient (no gradient computation needed)
- Follows the same methodology as sparse probing

❌ **NOT using**: Gradient attribution (expensive, not needed for k-sweep)

---

## Basic Usage

```bash
python train_probes.py \
    /path/to/sae/model \
    --sweep-k-attribution \
    --k-values 1 2 5 10 50 \
    --concepts domain-science sentiment-positive \
    --metrics mcc \
    --batch-size 28
```

### What This Does:

1. **Loads Data**: **Paired sentences** from `labeled_sentences_large_deduped_test.jsonl` using `load_paired_sentences_test_all_concepts()`
2. **Extracts SAE Activations**: Computes **delta SAE activations** (acts_2 - acts_1) for difference-based SAEs
3. **Extracts Residual Activations**: Computes **delta residual stream activations** (residual_acts_2 - residual_acts_1) from both sentences
4. **Trains Change Detector Probes**: Binary LR probes trained to detect concept changes between sentence pairs (one probe per concept)
5. **Ranks Features**: Using `|probe.coef_[0]|` (absolute LR coefficients) for each concept
6. **Sweeps K Values**: Tests probe performance with top-1, top-2, top-5, top-10, top-50 features
7. **Computes Metrics**:
   - Probe correlation with labels
   - Test accuracy
   - MCC (Matthews Correlation Coefficient)

---

## Command-Line Arguments

### Required
- `model_path`: Path to trained SAE model directory

### K-Sweep Specific
- `--sweep-k-attribution`: Enable k-sweep mode (uses LR coefficients, not gradient attribution)
- `--k-values`: List of k values to test (default: `1 2 5 10 50`)
- `--concepts`: Specific concepts to evaluate (e.g., `domain-science sentiment-positive`)
  - If not specified, evaluates ALL concepts

### Optional
- `--metrics`: Metrics to compute (default: `mcc`)
- `--batch-size`: Batch size for embedding extraction (default: 128)
- `--seed`: Random seed (default: 42)
- `--max-samples`: Limit number of samples for faster testing
- `--sae-activations-path`: Path to pre-computed SAE activations (`.npy` or `.pt`)
- `--plot-output`: Save plots to file (e.g., `mcc_vs_k.png`)
- `--mcc-union-features`: Use union of top-k features across concepts for MCC

### Model Configuration (Auto-inferred if not provided)
- `--lm-model-name`: Language model name (e.g., `EleutherAI/pythia-70m-deduped`)
- `--submodule-steer`: Submodule for SAE steering (e.g., `gpt_neox.layers.5`)
- `--submodule-probe`: Submodule for probe (defaults to same as `--submodule-steer`)

---

## Examples

### Example 1: Basic K-Sweep on Two Concepts
```bash
python train_probes.py \
    /network/scratch/j/joshi.shruti/ssae/labeled-sentences/paper_models/labeled-sentences_gemma-2-2b-it_seed0 \
    --sweep-k-attribution \
    --concepts domain-science sentiment-positive \
    --k-values 1 2 5 10 50 \
    --batch-size 28
```

**Output**:
```
K-SWEEP WITH LR COEFFICIENT-BASED FEATURE SELECTION (DIFFERENCE-BASED SAE)
======================================================================
Loading paired sentences from labeled_sentences_large_deduped_test.jsonl...
Loaded 10000 pairs for 11 concepts

Computing delta SAE activations (acts_2 - acts_1)...
Extracting embeddings for first sentences...
Extracting embeddings for second sentences...
Delta SAE activations shape: (10000, 65536)

Extracting residual stream activations from language model (paired sentences)...
Extracting residual activations for first sentences...
Extracting residual activations for second sentences...
Delta residual stream activations shape: (8000, 2048)

Training initial change detector probes for attribution...
  domain-science: Train acc = 0.8234
  sentiment-positive: Train acc = 0.7845

Extracting LR feature scores from trained probes...
  domain-science...
  sentiment-positive...

Sweeping k values: [1, 2, 5, 10, 50]
MCC computation method: per-concept

k =   1:
  domain-science:
    Probe Correlation: 0.8234
    Test Acc: 0.9012
    MCC: 0.8156
  sentiment-positive:
    Probe Correlation: 0.7845
    Test Acc: 0.8756
    MCC: 0.7821
  Aggregate Activation MCC: 0.7988

k =   2:
  ...
```

### Example 2: All Concepts with Plot
```bash
python train_probes.py \
    /path/to/sae/model \
    --sweep-k-attribution \
    --k-values 1 3 5 10 25 50 100 \
    --plot-output results/mcc_vs_k.png
```

### Example 3: Using Pre-computed SAE Activations
```bash
python train_probes.py \
    /path/to/sae/model \
    --sweep-k-attribution \
    --sae-activations-path /path/to/precomputed_activations.npy \
    --k-values 1 2 5 10 50 \
    --concepts tense-present tense-past
```

### Example 4: MCC with Union Features
```bash
python train_probes.py \
    /path/to/sae/model \
    --sweep-k-attribution \
    --k-values 5 10 50 \
    --mcc-union-features \
    --concepts domain-science domain-fantasy sentiment-positive sentiment-negative
```

**What `--mcc-union-features` does**:
- Takes the UNION of top-k features across all concepts
- Computes aggregate MCC using the combined feature set
- Alternative: per-concept (default) computes MCC per concept and averages

---

## Implementation Details

### 1. Load Paired Sentences (Lines 3434-3448)
```python
# Load paired sentences for all concepts
sentences_1, sentences_2, labels = load_paired_sentences_test_all_concepts(
    primary_concept=("tense", "present"), seed=args.seed
)
# Returns unified pairs with labels for ALL concepts
```

### 2. Compute Delta SAE Activations (Lines 3454-3485)
```python
# Extract embeddings from language model for both sentences
embeddings_1 = get_sentence_embeddings(sentences_1, sae_model.model_name, sae_model.layer, args.batch_size)
embeddings_2 = get_sentence_embeddings(sentences_2, sae_model.model_name, sae_model.layer, args.batch_size)

# Get SAE activations for both
sae_acts_1 = get_activations(sae_model, embeddings_1).numpy()
sae_acts_2 = get_activations(sae_model, embeddings_2).numpy()

# Compute delta for difference-based SAEs
sae_activations = sae_acts_2 - sae_acts_1  # Delta activations
```

### 3. Compute Delta Residual Activations (Lines 3556-3615)
```python
# Extract residual activations from BOTH sentences
for batch_1 in sentences_1_train:
    batch_tokenized_1 = tokenizer(batch_1, return_tensors="pt", padding=True).to(lm_model.device)
    _ = lm_model(**batch_tokenized_1)  # Captured by hook

for batch_2 in sentences_2_train:
    batch_tokenized_2 = tokenizer(batch_2, return_tensors="pt", padding=True).to(lm_model.device)
    _ = lm_model(**batch_tokenized_2)  # Captured by hook

# Compute delta residual activations
residual_acts_train = (residual_acts_2 - residual_acts_1).numpy()
```

### 4. Train Change Detector Probes (Lines 3626-3641)
```python
# Train binary probes on delta residual activations
for concept in labels_train_dict.keys():
    probe = LogisticRegression(random_state=args.seed, max_iter=1000, class_weight="balanced")
    probe.fit(residual_acts_train_normalized, labels_train_dict[concept])
    initial_probes[concept] = probe
```

### 5. Extract LR Coefficient Scores (Lines 3643-3653)
```python
# For each concept, extract LR coefficients from trained change detector probe
probe = initial_probes[concept]
all_scores = np.abs(probe.coef_[0])  # Absolute value of coefficients
all_scores_dict[concept] = all_scores
```

### 6. Sweep K Values (Lines 1659-1670)
```python
# For each k value, select top-k features and train sparse probe
all_scores = all_feature_scores_dict[concept]
_, top_k_indices = t.topk(all_scores, k=k)

result = train_sparse_probe_on_top_k(
    sae_activations_train,  # Delta SAE activations
    sae_activations_test,
    labels_train,
    labels_test,
    top_k_indices,
    seed,
    verbose=False,
)
```

---

## Why LR Coefficients (Not Gradient Attribution)?

### K-Sweep Uses BOTH Pairwise Data AND LR Coefficients:
- **Pairwise data**: For difference-based SAEs (delta_x = x2 - x1)
- **LR coefficient ranking**: Fast feature selection from trained probes
- This combines the best of both approaches: proper delta computation + efficient ranking

### Advantages of LR Coefficient Method:
✅ **Fast**: No need to run forward/backward passes through LLM for feature ranking
✅ **Simple**: Directly from trained probe weights
✅ **Effective**: Matches `heuristic_feature_ranking_binary` with `method="lr"`
✅ **Interpretable**: Feature importance = |coefficient|
✅ **Compatible with paired data**: Works seamlessly with change detector probes

### When Gradient Attribution IS Still Needed:
- For finding features for **causal interventions** with full gradient path through SAE
- When you need attribution at the **SAE latent level** (not residual level)
- For more sophisticated attribution methods beyond probe coefficients

---

## Output Files

### Plots Generated (if `--plot-output` specified):
- Left panels: Per-concept probe correlation vs k
- Right panel: Aggregate activation-label MCC vs k

### Console Output:
- Per-concept metrics at each k value
- Probe correlation, test accuracy, MCC
- Aggregate MCC across all concepts

---

## Common Issues

### Issue 1: "No module named 'transformers'"
**Solution**: The k-sweep loads the LM model to extract residual stream activations
```bash
pip install transformers
```

### Issue 2: Out of Memory
**Solution**: Reduce batch size or limit samples
```bash
--batch-size 16 --max-samples 5000
```

### Issue 3: Different results each run
**Solution**: Set seed explicitly
```bash
--seed 42
```

---

## Related Functions

- `heuristic_feature_ranking_binary()`: Lines 524-541 - LR coefficient ranking
- `run_k_sweep_attribution()`: Lines 3415-3680 - Main k-sweep function
- `sweep_k_values_for_plots()`: Lines 1575-1659 - Core sweep logic
- `train_sparse_probe_on_top_k()`: Lines 1509-1573 - Train probe on selected features

---

## Notes on Implementation

### Argument Name
⚠️ **Argument name is historical**: `--sweep-k-attribution` originally used gradient attribution, but now uses LR coefficient-based selection for efficiency.

The name is kept for backward compatibility. The implementation is:
```python
all_scores = np.abs(probe.coef_[0])  # LR method, not gradient!
```

### Difference-Based SAE Approach
✅ **Paired sentences**: The implementation loads paired sentences and computes delta activations for both:
- **SAE activations**: `sae_acts_2 - sae_acts_1`
- **Residual activations**: `residual_acts_2 - residual_acts_1`

This ensures proper evaluation of difference-based SAEs that were trained on `delta_x = x2 - x1`.

### Data Loading
The function `load_paired_sentences_test_all_concepts()` is used to:
1. Establish unified pairs using a primary concept (e.g., `tense-present`)
2. Compute binary change labels (0/1) for ALL concepts on the same pairs
3. Ensure consistent evaluation across all concepts
