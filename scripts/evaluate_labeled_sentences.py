import torch as t
import h5py
import json
import argparse
from pathlib import Path
from typing import Dict, Tuple, Any, List, Optional
import yaml
import os
from dagma import utils
from dagma.linear import DagmaLinear
from dagma.nonlinear import DagmaMLP, DagmaNonlinear
from datasets import load_dataset
import numpy as np


def load_jsonl(filepath: str):
    with open(filepath, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_labeled_sentences_test():
    """Load test sentences with binary labels for each individual sentence (no pairing)."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    datapath = os.path.join(
        script_dir,
        "..",
        "data",
        "labeled-sentences",
        "labeled_sentences_large_deduped_test.jsonl",
    )
    if not os.path.exists(datapath):
        raise FileNotFoundError(f"Test file not found: {datapath}")

    # Load all labeled sentences
    labeled_sentences = load_jsonl(datapath)

    sentences = []
    all_labels = {}

    values_ordered = [
        "tense-present",
        "tense-past",
        "voice-active",
        "voice-passive",
        "domain-science",
        "domain-fantasy",
        "domain-news",
        "domain-other",
        "reading-level-high",
        "reading-level-low",
        "sentiment-positive",
        "sentiment-neutral",
        "sentiment-negative",
    ]

    # Initialize label collections
    for value in values_ordered:
        all_labels[value] = []

    # Create binary labels for each sentence
    for data in labeled_sentences:
        sentence = data["sentence"]
        sentences.append(sentence)

        binary_labels = {}

        # Initialize all binary labels to False
        for value in values_ordered:
            binary_labels[value] = False

        # Set binary labels based on attributes
        for key, val in data.items():
            if key == "sentence":
                continue
            if key == "reading_level":
                binary_labels["reading-level-high"] = val > 11.5
                binary_labels["reading-level-low"] = val <= 11.5
            else:
                kv = f"{key}-{val}"
                if kv in binary_labels:
                    binary_labels[kv] = True

        # Collect labels for each concept
        for concept in values_ordered:
            all_labels[concept].append(binary_labels[concept])

    return sentences, all_labels


def load_bias_in_bios():
    """Load bias-in-bios dataset with gender and profession labels."""
    print("Loading bias-in-bios dataset...")
    dataset = load_dataset("LabHC/bias_in_bios")
    test_data = dataset["test"]

    # Extract texts and labels
    texts = [item["hard_text"] for item in test_data]
    gender_labels = [item["gender"] for item in test_data]  # 0=male, 1=female
    profession_labels = [item["profession"] for item in test_data]

    print(f"Loaded {len(texts)} samples")
    print(f"Gender distribution: {sum(gender_labels)} female, {len(gender_labels) - sum(gender_labels)} male")

    # Get unique professions and create binary labels for each
    unique_professions = sorted(list(set(profession_labels)))
    print(f"Found {len(unique_professions)} unique professions: {unique_professions[:10]}...")

    # Create label dictionary
    all_labels = {}
    all_labels["gender"] = gender_labels

    # Create binary labels for each profession
    for profession in unique_professions:
        all_labels[f"profession-{profession}"] = [
            1 if prof == profession else 0 for prof in profession_labels
        ]

    return texts, all_labels, unique_professions


def score_identification(acts, labels, lamda=0.1, metric="accuracy"):
    scores = {}
    top_features = {}
    labels = {
        k: v
        for k, v in labels.items()
        if k
        not in (
            "formality-high",
            "formality-neutral",
            "reading-level-low",
            "reading-level-high",
        )
    }
    label_matrix = t.stack(
        [t.Tensor(labels[l]) for l in labels], dim=0
    )  # N x L

    for label_name in labels:
        if metric == "mcc":
            label_vec = t.Tensor(labels[label_name])  # N
        else:
            label_vec = t.tensor(labels[label_name])
        feature_labels = acts.T > lamda  # F x N
        if metric == "accuracy":
            matches = feature_labels == label_vec
            accuracies = matches.sum(dim=1) / label_vec.shape[-1]
            accuracy = accuracies.max()
            top_features[label_name] = accuracies.argmax()
            scores[label_name] = accuracy
        elif metric == "macrof1":
            # Calculate true positives, false positives, false negatives for each feature
            true_positives = (
                (feature_labels & label_vec).sum(dim=1).float()
            )  # F
            false_positives = (
                (feature_labels & ~label_vec).sum(dim=1).float()
            )  # F
            false_negatives = (
                (~feature_labels & label_vec).sum(dim=1).float()
            )  # F

            # Calculate precision and recall
            precision = true_positives / (
                true_positives + false_positives + 1e-10
            )  # F
            recall = true_positives / (
                true_positives + false_negatives + 1e-10
            )  # F

            # Calculate F1 scores
            f1_scores = (
                2 * precision * recall / (precision + recall + 1e-10)
            )  # F

            # Find the feature with the max F1 score
            top_feature = f1_scores.argmax()
            max_f1 = f1_scores[top_feature]

            top_features[label_name] = top_feature
            scores[label_name] = max_f1
        elif metric == "mcc":
            acts_centered = acts - acts.mean(dim=0, keepdim=True)
            acts_std = acts_centered.norm(dim=0, keepdim=True)
            # Convert boolean labels to float for mean calculation
            label_matrix_float = label_matrix.T.float()
            label_matrix_centered = (
                label_matrix_float
                - label_matrix_float.mean(dim=0, keepdim=True)
            )
            label_matrix_std = label_matrix_centered.norm(dim=0, keepdim=True)
            # Correct correlation computation
            numerator = acts_centered.T @ label_matrix_centered  # F × L
            denominator = acts_std.T * label_matrix_std  # F × L (broadcasting)

            mask = denominator != 0  # prevent NaNs
            corr_matrix = t.zeros_like(numerator)
            corr_matrix[mask] = numerator[mask] / denominator[mask]

            # Get indices of maximum correlations for each label
            top_feature_indices = corr_matrix.argmax(
                dim=0
            )  # Returns indices, shape: (L,)
            top_features = {
                label_name: top_feature_indices[i].item()
                for i, label_name in enumerate(list(labels))
            }

            return corr_matrix, top_features
        else:
            raise ValueError(f"Unrecognized metric: {metric}")

    return scores, top_features


def identify_concept_dimensions(
    activations: t.Tensor,
    labels: Dict[str, List],
    top_n: int = 1,
    correlation_threshold: float = 0.3,
) -> Tuple[Dict[int, List[Tuple[str, float]]], Dict[str, List[Tuple[int, float]]], t.Tensor]:
    """
    Identify which latent dimensions correspond to which concepts based on correlation.

    Args:
        activations: Tensor of shape (N, F) where N is number of samples, F is number of features
        labels: Dictionary mapping concept names to binary labels (lists of 0/1)
        top_n: Number of top correlated concepts to assign to each dimension
        correlation_threshold: Minimum correlation to consider

    Returns:
        Tuple of:
        - dim_to_concepts: Dictionary mapping dimension index -> list of (concept_name, correlation) tuples
        - label_to_dims: Dictionary mapping label name -> list of (dimension_index, correlation) tuples
        - corr_matrix: Full correlation matrix (F x C)
    """
    print(f"Identifying concept dimensions...")
    print(f"Activations shape: {activations.shape}")
    print(f"Number of concepts: {len(labels)}")

    # Convert labels to tensor matrix
    label_names = list(labels.keys())
    label_matrix = t.stack([t.tensor(labels[name], dtype=t.float32) for name in label_names], dim=1)  # N x C

    # Center activations and labels
    acts_centered = activations - activations.mean(dim=0, keepdim=True)
    acts_std = acts_centered.norm(dim=0, keepdim=True)

    labels_centered = label_matrix - label_matrix.mean(dim=0, keepdim=True)
    labels_std = labels_centered.norm(dim=0, keepdim=True)

    # Compute correlation matrix: F x C
    numerator = acts_centered.T @ labels_centered  # F × C
    denominator = acts_std.T * labels_std  # F × C (broadcasting)

    mask = denominator != 0
    corr_matrix = t.zeros_like(numerator)
    corr_matrix[mask] = numerator[mask] / denominator[mask]

    # For each dimension, find top correlated concepts
    dim_to_concepts = {}
    n_features = activations.shape[1]

    for dim_idx in range(n_features):
        dim_corrs = corr_matrix[dim_idx]  # C
        abs_corrs = dim_corrs.abs()

        # Get top-n concepts above threshold
        top_indices = abs_corrs.argsort(descending=True)[:top_n]
        concept_list = []

        for idx in top_indices:
            corr_val = dim_corrs[idx].item()
            if abs(corr_val) >= correlation_threshold:
                concept_name = label_names[idx]
                concept_list.append((concept_name, corr_val))

        if concept_list:
            dim_to_concepts[dim_idx] = concept_list

    # For each label, find all dimensions above threshold AND ensure top dim is included
    label_to_dims = {}
    n_concepts = len(label_names)

    for concept_idx in range(n_concepts):
        concept_name = label_names[concept_idx]
        concept_corrs = corr_matrix[:, concept_idx]  # F
        abs_corrs = concept_corrs.abs()

        # Get top dimension for this label (guaranteed inclusion)
        top_dim_idx = abs_corrs.argmax().item()
        top_corr_val = concept_corrs[top_dim_idx].item()

        # Find all dimensions above threshold
        above_threshold_mask = abs_corrs >= correlation_threshold
        above_threshold_indices = above_threshold_mask.nonzero(as_tuple=True)[0]

        # Combine: ensure top dim is included, plus all above threshold
        dim_set = set(above_threshold_indices.tolist())
        dim_set.add(top_dim_idx)  # Ensure top dim is always included

        # Sort by absolute correlation (descending)
        dim_list = sorted(
            [(dim_idx, concept_corrs[dim_idx].item()) for dim_idx in dim_set],
            key=lambda x: abs(x[1]),
            reverse=True
        )

        label_to_dims[concept_name] = dim_list

    print(f"Found {len(dim_to_concepts)} dimensions with correlation >= {correlation_threshold}")

    # Print label statistics
    print(f"\nLabel dimension statistics:")
    for label_name in sorted(label_to_dims.keys())[:10]:  # Show first 10
        dims = label_to_dims[label_name]
        print(f"  {label_name}: {len(dims)} dimensions (top: dim {dims[0][0]}, corr={dims[0][1]:.3f})")
    if len(label_to_dims) > 10:
        print(f"  ... and {len(label_to_dims) - 10} more labels")

    return dim_to_concepts, label_to_dims, corr_matrix


def learn_latent_dag(
    activations: t.Tensor,
    n_samples: int = 500,
    lambda1: float = 0.02,
    top_k: int = 10,
    dim_labels: Optional[Dict[int, List[Tuple[str, float]]]] = None,
    top_dims: Optional[int] = 100,
    specific_dims: Optional[List[int]] = None,
) -> t.Tensor:
    """
    Learn a DAG on the difference between activations.

    Args:
        activations: Tensor of shape (N, F) where N is number of samples, F is number of features
        n_samples: Number of pairs to sample for computing differences
        lambda1: Regularization parameter for DAGMA
        top_k: Number of top connections to print
        dim_labels: Optional dictionary mapping dimension index to list of (concept_name, correlation) tuples
        top_dims: Number of top dimensions to use (based on variance). None = use all dims
        specific_dims: Optional list of specific dimension indices to use (overrides top_dims)

    Returns:
        W_est: Estimated coefficient matrix (F x F) or (subset x subset)
    """
    import numpy as np

    n_total = activations.shape[0]
    n_features = activations.shape[1]

    print(f"Learning DAG on activations with shape: {activations.shape}")

    # Select dimensions: prefer specific_dims if provided, otherwise use variance-based selection
    if specific_dims is not None:
        print(f"Using {len(specific_dims)} specific dimensions (union of top + above threshold)...")
        top_dim_indices_sorted = t.tensor(specific_dims, dtype=t.long)
        activations_subset = activations[:, top_dim_indices_sorted]

        # Create mapping from subset index to original index
        subset_to_original = {i: idx for i, idx in enumerate(specific_dims)}
        original_to_subset = {idx: i for i, idx in enumerate(specific_dims)}

        print(f"Reduced to shape: {activations_subset.shape}")
        print(f"Selected dims (first 20): {specific_dims[:20]}...")

        # Update dim_labels to use subset indices
        if dim_labels is not None:
            dim_labels_subset = {}
            for orig_idx, labels in dim_labels.items():
                if orig_idx in original_to_subset:
                    dim_labels_subset[original_to_subset[orig_idx]] = labels
            dim_labels = dim_labels_subset
            print(f"Retained {len(dim_labels)} labeled dimensions in subset")
    elif top_dims is not None and top_dims < n_features:
        print(f"Selecting top {top_dims} dimensions by variance for efficiency...")
        variances = activations.var(dim=0)
        top_dim_indices = variances.argsort(descending=True)[:top_dims]
        top_dim_indices_sorted = top_dim_indices.sort().values
        activations_subset = activations[:, top_dim_indices_sorted]

        # Create mapping from subset index to original index
        subset_to_original = {i: idx.item() for i, idx in enumerate(top_dim_indices_sorted)}
        original_to_subset = {idx.item(): i for i, idx in enumerate(top_dim_indices_sorted)}

        print(f"Reduced to shape: {activations_subset.shape}")
        print(f"Top variance dims: {top_dim_indices_sorted[:10].tolist()}...")

        # Update dim_labels to use subset indices
        if dim_labels is not None:
            dim_labels_subset = {}
            for orig_idx, labels in dim_labels.items():
                if orig_idx in original_to_subset:
                    dim_labels_subset[original_to_subset[orig_idx]] = labels
            dim_labels = dim_labels_subset
            print(f"Retained {len(dim_labels)} labeled dimensions in subset")
    else:
        activations_subset = activations
        subset_to_original = {i: i for i in range(n_features)}
        top_dim_indices_sorted = None

    n_features_subset = activations_subset.shape[1]

    print(f"Sampling {n_samples} pairs uniformly...")

    # Uniformly sample pairs of activations
    indices_i = t.randint(0, n_total, (n_samples,))
    indices_j = t.randint(0, n_total, (n_samples,))

    # Compute differences
    differences = activations_subset[indices_i] - activations_subset[indices_j]
    print(f"Differences shape: {differences.shape}")

    # Convert to numpy for DAGMA
    X = differences.cpu().numpy()

    # Learn DAG using DAGMA Linear with L2 loss
    print(f"Fitting DAGMA with lambda1={lambda1} on {n_features_subset} dimensions...")
    model = DagmaLinear(loss_type="l2")
    W_est = model.fit(X, lambda1=lambda1)

    print(f"DAG learning complete. Coefficient matrix shape: {W_est.shape}")

    # Find connections with highest coefficients
    W_abs = np.abs(W_est)

    # Get indices of top-k connections
    flat_indices = np.argsort(W_abs.flatten())[::-1][:top_k]
    top_connections = [
        (idx // n_features_subset, idx % n_features_subset, W_abs.flatten()[idx])
        for idx in flat_indices
    ]

    # Helper function to format dimension labels
    def format_dim_label(subset_dim_idx):
        orig_dim_idx = subset_to_original[subset_dim_idx]
        if dim_labels is None or subset_dim_idx not in dim_labels:
            return f"Dim {orig_dim_idx}"
        concepts = dim_labels[subset_dim_idx]
        concept_str = ", ".join([f"{name} ({corr:.2f})" for name, corr in concepts])
        return f"Dim {orig_dim_idx} [{concept_str}]"

    print(f"\nTop {top_k} connections (arrows) in the learned DAG:")
    print("=" * 80)
    for from_dim, to_dim, coef in top_connections:
        if coef > 0:  # Only show non-zero connections
            from_label = format_dim_label(from_dim)
            to_label = format_dim_label(to_dim)
            print(f"{from_label} -> {to_label}: {coef:.4f}")

    # Print summary of labeled dimensions involved in top connections
    if dim_labels is not None:
        involved_dims = set()
        for from_dim, to_dim, _ in top_connections:
            if from_dim in dim_labels:
                involved_dims.add(from_dim)
            if to_dim in dim_labels:
                involved_dims.add(to_dim)

        if involved_dims:
            print(f"\n{len(involved_dims)} labeled dimensions involved in top connections:")
            print("=" * 80)
            for subset_dim_idx in sorted(involved_dims):
                orig_dim_idx = subset_to_original[subset_dim_idx]
                concepts = dim_labels[subset_dim_idx]
                concept_str = ", ".join([f"{name} ({corr:.2f})" for name, corr in concepts])
                print(f"  Dim {orig_dim_idx}: {concept_str}")

    return t.from_numpy(W_est)


def load_model(model_path: Path):
    """Load trained SSAE model."""
    from ssae import DictLinearAE

    # Load model config
    config_path = model_path / "cfg.yaml"
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    # We need rep_dim - assume it's available in the config or load from embedding config
    # For now, we'll get it from the model weights
    weights_path = model_path / "weights.pth"
    state_dict = t.load(weights_path, map_location="cpu")

    # Get dimensions from the saved weights
    rep_dim = state_dict["encoder.weight"].shape[1]  # input dimension
    hid_dim = state_dict["encoder.weight"].shape[0]  # hidden dimension

    # Create model (assume layer norm)
    model = DictLinearAE(rep_dim, hid_dim, cfg.get("norm", "ln"))
    model.load_state_dict(state_dict)
    model.eval()

    return model


def get_sentence_embeddings(
    sentences,
    model_name="EleutherAI/pythia-70m-deduped",
    layer=5,
    batch_size=128,
):
    """Extract embeddings for individual sentences using store_embeddings function."""
    import sys
    import os

    # Add the parent directory to sys.path to import store_embeddings
    current_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(current_dir)

    from ssae.store_embeddings import extract_embeddings
    import torch
    from transformers import GPTNeoXForCausalLM, AutoTokenizer, AutoModelForCausalLM

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the language model
    if model_name == "EleutherAI/pythia-70m-deduped":
        model = GPTNeoXForCausalLM.from_pretrained(
            "EleutherAI/pythia-70m-deduped",
            revision="step3000",
            cache_dir="./pythia-70m-deduped/step3000",
        ).to(device)
        tokenizer = AutoTokenizer.from_pretrained(
            "EleutherAI/pythia-70m-deduped",
            revision="step3000",
            cache_dir="./pythia-70m-deduped/step3000",
        )
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
    elif model_name == "google/gemma-2-2b-it":
        model = AutoModelForCausalLM.from_pretrained(
            model_name, low_cpu_mem_usage=True
        ).to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
    else:
        raise NotImplementedError(f"Model {model_name} not supported")

    # Convert individual sentences to fake pairs format for extract_embeddings
    # We'll duplicate each sentence so we get (sentence, sentence) pairs
    fake_contexts = [(sentence, sentence) for sentence in sentences]

    # Extract embeddings using the existing function
    embeddings = extract_embeddings(
        fake_contexts, model, tokenizer, layer, "last_token", batch_size
    )

    # Since we used (sentence, sentence) pairs, the difference will be zero
    # So we just take the first embedding from each pair
    return embeddings[:, 0, :]  # Shape: (N, rep_dim)


def get_activations(
    model: t.nn.Module, embeddings: t.Tensor, batch_size: int = 512
) -> t.Tensor:
    """Get SSAE activations for embeddings."""
    model.eval()
    all_activations = []

    with t.no_grad():
        for i in range(0, len(embeddings), batch_size):
            batch = embeddings[i : i + batch_size]
            _, activations = model(batch)  # Get hidden activations
            all_activations.append(activations.cpu())

    return t.cat(all_activations, dim=0)


def evaluate_sentence_labels(
    model_path: Path,
    threshold: float = 0.1,
    metrics: list = ["accuracy", "macrof1", "mcc"],
) -> Dict[str, Any]:
    """Evaluate SSAE on individual sentence labels."""

    # Load test sentences and labels
    sentences, labels = load_labeled_sentences_test()
    print(f"Loaded {len(sentences)} test sentences")
    print(f"Available labels: {list(labels.keys())}")

    # Get sentence embeddings
    print("Extracting sentence embeddings...")
    embeddings = get_sentence_embeddings(sentences)
    print(f"Embeddings shape: {embeddings.shape}")

    # Load model
    model = load_model(model_path)
    print(f"Loaded model from {model_path}")

    # Get SSAE activations
    print("Getting SSAE activations...")
    activations = get_activations(model, embeddings)
    print(f"Activations shape: {activations.shape}")

    # Convert labels to tensors
    tensor_labels = {}
    for concept, values in labels.items():
        tensor_labels[concept] = t.tensor(values, dtype=t.bool)

    # Evaluate for each metric
    results = {}
    for metric in metrics:
        print(f"Evaluating {metric}...")
        if metric == "mcc":
            corr_matrix, top_features = score_identification(
                activations, tensor_labels, threshold, metric
            )
            results[metric] = {
                "correlation_matrix": corr_matrix,
                "top_features": top_features,
            }
        else:
            scores, top_features = score_identification(
                activations, tensor_labels, threshold, metric
            )
            results[metric] = {"scores": scores, "top_features": top_features}

    return results


def evaluate_bias_in_bios_dag(
    model_path: Path,
    n_samples: int = 500,
    lambda1: float = 0.02,
    top_k: int = 20,
    correlation_threshold: float = 0.3,
    max_texts: Optional[int] = None,
    embedding_model: str = "EleutherAI/pythia-70m-deduped",
    layer: int = 5,
    embedding_batch_size: int = 128,
    activation_batch_size: int = 512,
    top_dims: Optional[int] = 100,
) -> Dict[str, Any]:
    """
    Evaluate bias-in-bios dataset and learn DAG with labeled dimensions.

    Args:
        model_path: Path to trained SSAE model
        n_samples: Number of pairs to sample for DAG learning
        lambda1: DAGMA regularization parameter
        top_k: Number of top connections to display
        correlation_threshold: Minimum correlation for dimension labeling
        max_texts: Optional limit on number of texts to process (for faster testing)
        embedding_model: Model to use for extracting embeddings
        layer: Layer to extract embeddings from
        embedding_batch_size: Batch size for embedding extraction
        activation_batch_size: Batch size for SSAE activation extraction
        top_dims: Number of top dimensions to use for DAG learning (None = use all)

    Returns:
        Dictionary containing results including DAG weights and dimension labels
    """
    # Load bias-in-bios dataset
    texts, labels, professions = load_bias_in_bios()

    # Optionally limit dataset size for faster testing
    if max_texts is not None and max_texts < len(texts):
        print(f"Limiting to first {max_texts} texts for faster processing...")
        texts = texts[:max_texts]
        labels = {k: v[:max_texts] for k, v in labels.items()}

    # Get sentence embeddings
    print(f"\nExtracting sentence embeddings using {embedding_model} layer {layer}...")
    print(f"Embedding batch size: {embedding_batch_size}")
    embeddings = get_sentence_embeddings(texts, model_name=embedding_model, layer=layer, batch_size=embedding_batch_size)
    print(f"Embeddings shape: {embeddings.shape}")

    # Load model
    model = load_model(model_path)
    print(f"Loaded model from {model_path}")

    # Get SSAE activations
    print("\nGetting SSAE activations...")
    print(f"Activation batch size: {activation_batch_size}")
    activations = get_activations(model, embeddings, batch_size=activation_batch_size)
    print(f"Activations shape: {activations.shape}")

    # Identify which dimensions correspond to which concepts
    print("\n" + "=" * 80)
    print("IDENTIFYING CONCEPT DIMENSIONS")
    print("=" * 80)
    dim_labels, label_to_dims, corr_matrix = identify_concept_dimensions(
        activations, labels, top_n=2, correlation_threshold=correlation_threshold
    )

    # Print identified concept dimensions
    print(f"\nLabeled dimensions (correlation >= {correlation_threshold}):")
    for dim_idx in sorted(dim_labels.keys())[:20]:  # Show first 20
        concepts = dim_labels[dim_idx]
        concept_str = ", ".join([f"{name} ({corr:.2f})" for name, corr in concepts])
        print(f"  Dim {dim_idx}: {concept_str}")
    if len(dim_labels) > 20:
        print(f"  ... and {len(dim_labels) - 20} more dimensions")

    # Print label-to-dimensions mapping
    print(f"\nLabel-to-dimensions mapping (each label's dimensions):")
    for label_name in sorted(label_to_dims.keys())[:20]:  # Show first 20
        dims = label_to_dims[label_name]
        dims_str = ", ".join([f"dim {dim_idx} ({corr:.2f})" for dim_idx, corr in dims[:5]])  # Show top 5 dims per label
        if len(dims) > 5:
            dims_str += f" ... and {len(dims) - 5} more"
        print(f"  {label_name}: {len(dims)} dims - {dims_str}")
    if len(label_to_dims) > 20:
        print(f"  ... and {len(label_to_dims) - 20} more labels")

    # Collect union of all relevant dimensions for DAG learning
    # Union = top dim for each label + all dims above threshold
    relevant_dims = set()
    for label_name, dims_list in label_to_dims.items():
        for dim_idx, corr in dims_list:
            relevant_dims.add(dim_idx)

    relevant_dims_sorted = sorted(list(relevant_dims))
    print(f"\nTotal relevant dimensions for DAG learning: {len(relevant_dims_sorted)}")
    print(f"  (Union of top dims per label + all dims above threshold {correlation_threshold})")

    # Learn DAG on activation differences with labeled dimensions
    print("\n" + "=" * 80)
    print("LEARNING DAG ON ACTIVATION DIFFERENCES")
    print("=" * 80)
    W_est = learn_latent_dag(
        activations,
        n_samples=n_samples,
        lambda1=lambda1,
        top_k=top_k,
        dim_labels=dim_labels,
        top_dims=top_dims,
        specific_dims=relevant_dims_sorted,
    )

    return {
        "W_est": W_est,
        "dim_labels": dim_labels,
        "label_to_dims": label_to_dims,
        "corr_matrix": corr_matrix,
        "relevant_dims": relevant_dims_sorted,
        "activations_shape": activations.shape,
        "n_labeled_dims": len(dim_labels),
        "n_relevant_dims": len(relevant_dims_sorted),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate SSAE on individual sentence labels or bias-in-bios with DAG learning"
    )
    parser.add_argument(
        "model_path", type=Path, help="Path to trained model directory"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="labeled-sentences",
        choices=["labeled-sentences", "bias-in-bios"],
        help="Dataset to use for evaluation",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.1,
        help="Activation threshold for binarization (labeled-sentences only)",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=["accuracy", "macrof1"],
        choices=["accuracy", "macrof1", "mcc"],
        help="Metrics to compute (labeled-sentences only)",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=1000,
        help="Number of pairs to sample for DAG learning (bias-in-bios only)",
    )
    parser.add_argument(
        "--lambda1",
        type=float,
        default=0.02,
        help="DAGMA regularization parameter (bias-in-bios only)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="Number of top DAG connections to display (bias-in-bios only)",
    )
    parser.add_argument(
        "--correlation-threshold",
        type=float,
        default=0.3,
        help="Minimum correlation for dimension labeling (bias-in-bios only)",
    )
    parser.add_argument(
        "--max-texts",
        type=int,
        default=None,
        help="Maximum number of texts to process (bias-in-bios only, for faster testing)",
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default="EleutherAI/pythia-70m-deduped",
        choices=["EleutherAI/pythia-70m-deduped", "google/gemma-2-2b-it"],
        help="Model to use for extracting embeddings",
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=5,
        help="Layer to extract embeddings from",
    )
    parser.add_argument(
        "--embedding-batch-size",
        type=int,
        default=128,
        help="Batch size for embedding extraction (reduce if OOM)",
    )
    parser.add_argument(
        "--activation-batch-size",
        type=int,
        default=512,
        help="Batch size for SSAE activation extraction (reduce if OOM)",
    )
    parser.add_argument(
        "--top-dims",
        type=int,
        default=100,
        help="Number of top dimensions to use for DAG learning (bias-in-bios only, for speed)",
    )
    parser.add_argument("--output", type=Path, help="Output file for results")

    args = parser.parse_args()

    # Evaluate model based on dataset choice
    if args.dataset == "bias-in-bios":
        results = evaluate_bias_in_bios_dag(
            args.model_path,
            n_samples=args.n_samples,
            lambda1=args.lambda1,
            top_k=args.top_k,
            correlation_threshold=args.correlation_threshold,
            max_texts=args.max_texts,
            embedding_model=args.embedding_model,
            layer=args.layer,
            embedding_batch_size=args.embedding_batch_size,
            activation_batch_size=args.activation_batch_size,
            top_dims=args.top_dims,
        )
        # Results are already printed by evaluate_bias_in_bios_dag
        return
    else:
        # Evaluate model with labeled sentences
        results = evaluate_sentence_labels(
            args.model_path, args.threshold, args.metrics
        )

    # Print results
    print("\n" + "=" * 70)
    print("SENTENCE-LEVEL EVALUATION RESULTS")
    print("=" * 70)

    for metric, data in results.items():
        print(f"\n{metric.upper()}:")
        if metric == "mcc":
            scores = data["correlation_matrix"]
            top_features = data["top_features"]
            top_scores = scores.max(dim=0).values
            mcc = top_scores.mean().item()

            for i, label in enumerate(list(top_features.keys())):
                if label not in ("domain-science", "sentiment-positive"):
                    continue
                print(f"{label}: {top_scores[i]} ({top_features[label]})")

            print(f"MCC: {mcc:.3f}")
            print(f"  Top features: {data['top_features']}")
            print(
                f"  Correlation matrix shape: {data['correlation_matrix'].shape}"
            )
        else:
            scores = data["scores"]
            top_features = data["top_features"]

            print(scores)
            print(sum(list(scores.values())))
            print(sum(list(scores.values())) / len(list(scores.keys())))
            print()
            print(top_features)

            for concept, score in scores.items():
                feature_idx = top_features[concept]
                print(f"  {concept}: {score:.4f} (feature {feature_idx})")

    # Save results if output specified
    if args.output:
        # Convert tensors to lists for JSON serialization
        save_results = {}
        save_results["dataset"] = "labeled-sentences"

        for metric, data in results.items():
            if metric == "mcc":
                scores = data["correlation_matrix"]
                top_features = data["top_features"]
                top_scores = scores.max(dim=0).values

                # Extract individual MCC scores for each concept
                mcc_scores = {}
                for i, label in enumerate(list(top_features.keys())):
                    mcc_scores[label] = float(top_scores[i])

                save_results[metric] = {
                    "scores": mcc_scores,
                    "average_mcc": float(top_scores.mean().item()),
                    "top_features": {
                        k: int(v) for k, v in data["top_features"].items()
                    },
                    "correlation_matrix_shape": list(
                        data["correlation_matrix"].shape
                    ),
                }
            else:
                save_results[metric] = {
                    "scores": {k: float(v) for k, v in data["scores"].items()},
                    "average_score": float(
                        sum(data["scores"].values()) / len(data["scores"])
                    ),
                    "top_features": {
                        k: int(v) for k, v in data["top_features"].items()
                    },
                }

        with open(args.output, "w") as f:
            json.dump(save_results, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
