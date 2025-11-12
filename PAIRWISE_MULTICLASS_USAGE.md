# Using PairwiseMulticlassProbingDataset for Difference-Based SAEs

## Overview

For SAEs trained on **differences** (delta_x = x2 - x1), we need to use **pairwise datasets** for training probes and gradient attribution.

---

## The Two Approaches

### 1. **Single Sentence Approach** (OLD - Incorrect for Difference SAEs)
```python
# Uses MulticlassProbingDataset
dataset = MulticlassProbingDataset(filepath, "domain")
# Returns: {"sentence": str, "label": int}
# label: 0=fantasy, 1=science, 2=news, 3=other

# Trains probe on: "which domain class is this sentence?"
```

### 2. **Pairwise Approach** (NEW - Correct for Difference SAEs)
```python
# Uses PairwiseMulticlassProbingDataset
dataset = PairwiseMulticlassProbingDataset(filepath, "domain", seed=42)
# Returns: {"sentence_1": str, "sentence_2": str, "label": int}
# label: 1 if domain changed, 0 if same

# Trains probe on: "did domain class change between s1 and s2?"
```

---

## Key Functions

### `prepare_pairwise_datasets_multiclass()`
```python
train_ds, test_ds = prepare_pairwise_datasets_multiclass(
    train_filepath="/path/to/train.jsonl",
    test_filepath="/path/to/test.jsonl",
    concept_key="domain",  # or "sentiment", "tense", "voice"
    seed=42
)

# Each item in dataset:
# {
#     "sentence_1": "The wizard cast a spell",      # fantasy
#     "sentence_2": "Scientists discovered a cure",  # science
#     "label": 1  # domain changed
# }
```

---

## Complete Workflow for `test_all_with_interventions`

### Step 1: Load Pairwise Datasets
```python
concepts = ["domain", "sentiment", "tense", "voice"]
train_filepath, test_filepath = get_filepaths(args.data_path, args.test_file)

# Load pairwise datasets for each concept GROUP
pairwise_train_data = {}
pairwise_test_data = {}

for concept in concepts:
    train_ds, test_ds = prepare_pairwise_datasets_multiclass(
        train_filepath, test_filepath, concept, args.seed
    )
    pairwise_train_data[concept] = train_ds
    pairwise_test_data[concept] = test_ds
```

### Step 2: Extract Delta Activations
```python
def extract_delta_activations_pairwise(model, dataloader, layer_num, model_type):
    """
    Extract delta activations from paired sentences.
    Returns: activations_2 - activations_1
    """
    activations_1 = []
    activations_2 = []
    labels = []

    for batch in dataloader:
        s1 = batch["sentence_1"]
        s2 = batch["sentence_2"]

        # Get activations for each sentence
        acts_1 = extract_activations_single(model, s1, layer_num, model_type)
        acts_2 = extract_activations_single(model, s2, layer_num, model_type)

        activations_1.append(acts_1)
        activations_2.append(acts_2)
        labels.extend(batch["label"])

    activations_1 = np.concatenate(activations_1, axis=0)
    activations_2 = np.concatenate(activations_2, axis=0)
    delta_activations = activations_2 - activations_1

    return delta_activations, np.array(labels)
```

### Step 3: Train Change Detector Probes
```python
change_detector_probes = {}

for concept in concepts:
    train_dataloader = DataLoader(pairwise_train_data[concept], batch_size=32)

    # Get delta activations
    delta_acts_train, labels_train = extract_delta_activations_pairwise(
        model, train_dataloader, args.layer_num, model_type
    )

    # Train binary change detector probe
    probe = LogisticRegression(random_state=args.seed, max_iter=1000)
    probe.fit(delta_acts_train, labels_train)

    change_detector_probes[concept] = probe
```

### Step 4: Find Top Features with Pairwise Gradient Attribution
```python
top_features = {}

for concept in concepts:
    train_ds = pairwise_train_data[concept]

    # Extract sentence pairs
    sentences_1 = [train_ds[i]["sentence_1"] for i in range(len(train_ds))]
    sentences_2 = [train_ds[i]["sentence_2"] for i in range(len(train_ds))]

    # Use pairwise gradient attribution
    top_scores, top_indices, all_scores = find_top_k_features_by_attribution_pairs(
        model=model._model,
        tokenizer=model.tokenizer,
        submodule_steer_name=f"gpt_neox.layers.{mid_layer}",
        submodule_probe_name=f"gpt_neox.layers.{args.layer_num}",
        dictionary=dictionary,
        probe=change_detector_probes[concept],
        sentences_1=sentences_1,
        sentences_2=sentences_2,
        k=1,
        use_sparsemax=False,
        batch_size=args.batch_size,
    )

    top_features[concept] = top_indices[0].item()
```

### Step 5: Build Intervention Matrix

Now we need to map from **multiclass concept groups** to **binary-level interventions**:

```python
# Map concept groups to binary concepts for intervention matrix
features_binary = {
    "domain": ["fantasy", "science", "news", "other"],
    "sentiment": ["positive", "neutral", "negative"],
    "voice": ["active", "passive"],
    "tense": ["present", "past"],
}

# Build matrix: rows = steering concepts (groups), cols = test concepts (binary)
score_matrix = []
IDX = {}

# Create index for binary concepts
idx = 0
for concept_group, values in features_binary.items():
    for value in values:
        IDX[f"{concept_group}-{value}"] = idx
        idx += 1

# For each multiclass concept group used for steering
for steer_concept in concepts:
    top_feature = top_features[steer_concept]
    feature_max = sae_acts[:, top_feature].max().item()

    row_scores = []

    # Test effect on each BINARY concept
    for test_concept, test_values in features_binary.items():
        for test_value in test_values:
            # Load single-sentence dataset for this binary concept
            _, test_dataset_binary = prepare_datasets(
                train_filepath, test_filepath,
                test_concept, test_value, args.seed
            )

            test_dataloader = DataLoader(test_dataset_binary, batch_size=32)

            # Extract original activations
            acts_org, labels = extract_activations(
                model, test_dataloader, -1, model_type
            )

            # Extract intervened activations (steer with group's top feature)
            acts_int = extract_activations_with_intervention(
                model, test_dataloader, dictionary, mid_layer, -1,
                feature_max, top_feature, model_type
            )

            # Load BINARY probe for this specific concept value
            probe_path = os.path.join(probes_dir, f"{test_concept}_{test_value}.joblib")
            probe_binary = joblib.load(probe_path)

            # Compare predictions
            logits_org = probe_binary.decision_function(acts_org)
            logits_int = probe_binary.decision_function(acts_int.to("cpu"))
            logits_delta = (logits_int - logits_org).mean()

            row_scores.append(logits_delta)

    score_matrix.append(row_scores)
```

---

## Key Differences Summary

| Aspect | Old (Single Sentence) | New (Pairwise) |
|--------|----------------------|----------------|
| **Dataset** | `MulticlassProbingDataset` | `PairwiseMulticlassProbingDataset` |
| **Input** | Single sentence | Pair of sentences |
| **Labels** | Class index (0,1,2,...) | Binary change (0 or 1) |
| **Probe Trains On** | "What class is this?" | "Did class change?" |
| **Activations** | Single: `acts` | Delta: `acts_2 - acts_1` |
| **Gradient Attribution** | `find_top_k_features_by_attribution` | `find_top_k_features_by_attribution_pairs` |
| **SAE Compatibility** | Single embedding SAEs | Difference-based SAEs ✓ |

---

## Matrix Interpretation

After using pairwise change detectors to find top features:

**Rows**: Multiclass concept groups (domain, sentiment, tense, voice)
**Columns**: Binary concept values (fantasy, science, positive, negative, ...)

**Matrix[i,j]**: How much does steering with concept group i's top feature affect binary concept j's probe logits?

Example:
- `Matrix["domain", "sentiment-positive"]`: Does steering with domain-change feature affect positivity detection?
