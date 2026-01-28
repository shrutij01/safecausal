import torch as t
import h5py
import json
import argparse
from pathlib import Path
from typing import Dict, Tuple, Any
import yaml
import os
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from scipy.stats import pearsonr
from tqdm import tqdm
import zipfile
import tempfile
import shutil
from functools import partial
from scripts.probe_data import (
    PairwiseProbingDataset,
    balance_dataset,
    concept_filter,
)

os.environ["HF_HOME"] = os.environ.get(
    "HF_HOME", "/network/scratch/j/joshi.shruti/hf_cache"
)
os.environ["HF_DATASETS_CACHE"] = os.environ.get(
    "HF_DATASETS_CACHE", "/network/scratch/j/joshi.shruti/hf_cache/datasets"
)


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


def load_pairwise_labeled_sentences_test(concept_key, concept_value, seed=0, max_pairs=None, split="test"):
    """
    Build a pairwise dataset from labeled_sentences for a binary concept.

    Creates balanced pairs where:
    - label=1: concept differs between sentence_1 and sentence_2 (change)
    - label=0: concept is the same in both sentences (no-change)

    This is direction-invariant: (pos→neg) and (neg→pos) both get label=1.

    Args:
        concept_key: e.g., "tense", "voice", "domain", "sentiment"
        concept_value: e.g., "present", "active", "science", "positive"
        seed: random seed for pair generation
        max_pairs: maximum number of pairs to generate (optional)
        split: "test" or "train" (determines which file to load)

    Returns:
        sentences_1: list[str] - first sentences in pairs
        sentences_2: list[str] - second sentences in pairs
        pair_labels: np.ndarray - 1 if concept changed, 0 if same
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))

    if split == "test":
        filename = "labeled_sentences_large_deduped_test.jsonl"
    elif split == "train":
        filename = "labeled_sentences_large_deduped_train.jsonl"
    else:
        raise ValueError(f"Unknown split '{split}'. Must be 'test' or 'train'.")

    datapath = os.path.join(
        script_dir,
        "..",
        "data",
        "labeled-sentences",
        filename,
    )
    if not os.path.exists(datapath):
        raise FileNotFoundError(f"File not found: {datapath}")

    # Use PairwiseProbingDataset to create pairs
    dataset = PairwiseProbingDataset(
        datapath,
        concept_key=concept_key,
        concept_value=concept_value,
        seed=seed
    )

    # Extract pairs
    sentences_1 = []
    sentences_2 = []
    pair_labels = []

    num_pairs = len(dataset)
    if max_pairs is not None:
        num_pairs = min(num_pairs, max_pairs)

    for i in range(num_pairs):
        item = dataset[i]
        sentences_1.append(item["sentence_1"])
        sentences_2.append(item["sentence_2"])
        pair_labels.append(item["label"])

    pair_labels = np.array(pair_labels, dtype=int)

    logging.info(
        f"Built {len(pair_labels)} pairs for {concept_key}='{concept_value}' "
        f"({pair_labels.sum()} change, {len(pair_labels) - pair_labels.sum()} no-change)."
    )

    return sentences_1, sentences_2, pair_labels


def load_paired_sentences_test_all_concepts(primary_concept=("tense", "present"), seed=42, max_pairs=None):
    """
    Load test sentences as unified pairs for difference-based SAE attribution.

    Uses PairwiseProbingDataset with a primary concept to establish one unified
    set of pairs, then computes labels for all concepts on those same pairs.

    This is useful for k-sweep experiments where you want the same pairs
    evaluated across multiple concepts.

    Args:
        primary_concept: tuple of (concept_key, concept_value) to use for creating pairs
        seed: random seed for pair generation
        max_pairs: maximum number of pairs (optional)

    Returns:
        sentences_1: list of first sentences in each pair
        sentences_2: list of second sentences in each pair
        all_labels: dict mapping concept name to binary labels (1=change, 0=no-change)
    """
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

    # Create primary dataset to establish unified pairs
    primary_key, primary_value = primary_concept
    primary_dataset = PairwiseProbingDataset(
        datapath,
        concept_key=primary_key,
        concept_value=primary_value,
        seed=seed
    )

    # Extract sentence pairs
    sentences_1 = []
    sentences_2 = []

    num_pairs = len(primary_dataset)
    if max_pairs is not None:
        num_pairs = min(num_pairs, max_pairs)

    for i in range(num_pairs):
        item = primary_dataset[i]
        sentences_1.append(item["sentence_1"])
        sentences_2.append(item["sentence_2"])

    # Now compute labels for all concepts on these same pairs
    # Load raw data to access all attributes
    labeled_sentences = load_jsonl(datapath)

    # Build sentence -> attributes mapping
    sent_to_attrs = {}
    for data in labeled_sentences:
        sent = data["sentence"]
        sent_to_attrs[sent] = data

    # Concepts to compute labels for
    concept_specs = [
        ("tense", "present"),
        ("tense", "past"),
        ("voice", "active"),
        ("voice", "passive"),
        ("domain", "science"),
        ("domain", "fantasy"),
        ("domain", "news"),
        ("domain", "other"),
        ("sentiment", "positive"),
        ("sentiment", "neutral"),
        ("sentiment", "negative"),
    ]

    all_labels = {}

    for concept_key, concept_value in concept_specs:
        concept_name = f"{concept_key}-{concept_value}"
        labels = []

        for s1, s2 in zip(sentences_1, sentences_2):
            attrs1 = sent_to_attrs.get(s1, {})
            attrs2 = sent_to_attrs.get(s2, {})

            # Check if sentence has this concept
            has_concept_1 = (concept_key in attrs1 and attrs1[concept_key] == concept_value)
            has_concept_2 = (concept_key in attrs2 and attrs2[concept_key] == concept_value)

            # Label: 1 if concept differs, 0 if same
            label = 1 if has_concept_1 != has_concept_2 else 0
            labels.append(label)

        all_labels[concept_name] = labels

    return sentences_1, sentences_2, all_labels


def prepare_datasets(
    train_filepath, test_filepath, concept_key, concept_value, seed=42
):
    """Prepare and balance the training and test datasets."""
    filter_criterion = partial(
        concept_filter, concept_key=concept_key, concept_value=concept_value
    )
    train_dataset = PairwiseProbingDataset(train_filepath, filter_criterion)
    test_dataset = PairwiseProbingDataset(test_filepath, filter_criterion)

    # print("Balancing training dataset...")
    train_dataset = balance_dataset(train_dataset, seed)
    # print("Balancing test dataset...")
    test_dataset = balance_dataset(test_dataset, seed)

    return train_dataset, test_dataset


def prepare_pairwise_datasets_multiclass(train_filepath, test_filepath, concept_key, seed=42):
    """
    Prepare pairwise datasets for multiclass concepts (e.g., domain, sentiment, tense, voice).

    Returns datasets where each item is a pair of sentences with:
    - label=1 if the concept CLASS differs between sentences (change)
    - label=0 if the concept CLASS is the same (no-change)

    This is for training "change detector" probes on multiclass concepts.

    Args:
        train_filepath: Path to training data JSONL
        test_filepath: Path to test data JSONL
        concept_key: One of "domain", "sentiment", "tense", "voice"
        seed: Random seed for pair generation

    Returns:
        train_dataset: PairwiseMulticlassProbingDataset
        test_dataset: PairwiseMulticlassProbingDataset

    Example:
        train_ds, test_ds = prepare_pairwise_datasets_multiclass(
            train_path, test_path, "domain", seed=42
        )
        # Each item: {"sentence_1": str, "sentence_2": str, "label": 0 or 1}
    """
    train_dataset = PairwiseMulticlassProbingDataset(
        train_filepath,
        concept=concept_key,
        seed=seed
    )
    test_dataset = PairwiseMulticlassProbingDataset(
        test_filepath,
        concept=concept_key,
        seed=seed + 1  # Different seed for test to avoid overlap
    )

    return train_dataset, test_dataset


def score_identification(
    acts,
    labels,
    lamda=0.1,
    metric="accuracy",
    probe=None,
    label_name=None,
    k=1,
    model=None,
    tokenizer=None,
    submodule_steer_name=None,
    submodule_probe_name=None,
    dictionary=None,
    sentences=None,
    use_sparsemax=False,
    batch_size=32,
):
    """
    Identify top features using different methods.

    Args:
        acts: Activations tensor (N x F)
        labels: Dictionary of label vectors or single label vector
        lamda: Threshold for binarization
        metric: One of "accuracy", "macrof1", "mcc", "gradient"
        probe: Trained probe (required for gradient method)
        label_name: Label name for gradient method (e.g., "domain-science")
        k: Number of top features to return (default: 1)
        model: Language model (required for gradient method with hooks)
        tokenizer: Tokenizer (required for gradient method with hooks)
        submodule_steer_name: Submodule name for SAE intervention (required for gradient)
        submodule_probe_name: Submodule name for probe activations (required for gradient)
        dictionary: SSAE dictionary (required for gradient method with hooks)
        sentences: List of text sentences (required for gradient method with hooks)
        use_sparsemax: Whether SAE uses sparsemax/MP activation (for gradient method)
        batch_size: Batch size for gradient attribution computation (default: 32)

    Returns:
        scores and top_features dict, or (corr_matrix, top_features) for mcc
        For gradient method with k>1: returns (scores_dict, top_features_dict) where
            scores_dict[label_name] = list of k scores
            top_features_dict[label_name] = list of k feature indices
    """
    if metric == "gradient":
        if probe is None or label_name is None:
            raise ValueError(
                "gradient method requires probe and label_name arguments"
            )

        required_params = [
            model,
            tokenizer,
            submodule_steer_name,
            submodule_probe_name,
            dictionary,
            sentences,
        ]
        if all(param is not None for param in required_params):
            top_k_scores, top_k_indices, all_scores = (
                find_top_k_features_by_attribution(
                    model=model,
                    tokenizer=tokenizer,
                    submodule_steer_name=submodule_steer_name,
                    submodule_probe_name=submodule_probe_name,
                    dictionary=dictionary,
                    probe=probe,
                    sentences=sentences,
                    k=k,
                    use_sparsemax=use_sparsemax,
                    batch_size=batch_size,
                )
            )

            # Format return values based on k
            if k == 1:
                return {label_name: top_k_scores[0].item()}, {
                    label_name: top_k_indices[0].item()
                }
            else:
                return (
                    {label_name: top_k_scores.tolist()},
                    {label_name: top_k_indices.tolist()},
                )
        else:
            raise ValueError(
                "gradient method requires: model, tokenizer, submodule_steer_name, "
                "submodule_probe_name, dictionary, and sentences parameters"
            )

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
            # Use centralized correlation computation
            corr_matrix, label_names = compute_correlation_matrix(acts, labels)

            # Get indices of maximum correlations for each label
            top_feature_indices = corr_matrix.argmax(
                dim=0
            )  # Returns indices, shape: (L,)
            top_features = {
                label_name: top_feature_indices[i].item()
                for i, label_name in enumerate(label_names)
            }

            return corr_matrix, top_features
        else:
            raise ValueError(f"Unrecognized metric: {metric}")

    return scores, top_features


def load_model(model_path: Path):
    """Load trained SSAE model and return model with config info."""
    from ssae import DictLinearAE

    # Check if model_path is a zip file
    temp_dir = None
    if model_path.suffix == ".zip":
        # Create temporary directory
        temp_dir = tempfile.mkdtemp()
        temp_path = Path(temp_dir)

        # Extract zip file
        with zipfile.ZipFile(model_path, "r") as zip_ref:
            zip_ref.extractall(temp_path)

        # Update model_path to the extracted directory
        # Assuming the zip contains a single directory or files at root
        extracted_items = list(temp_path.iterdir())
        if len(extracted_items) == 1 and extracted_items[0].is_dir():
            model_path = extracted_items[0]
        else:
            model_path = temp_path

    try:
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

        # Extract model name and layer from config
        model_name = cfg.get("extra", {}).get(
            "model", "EleutherAI/pythia-70m-deduped"
        )
        layer = cfg.get("extra", {}).get("llm_layer", 5)

        # Create model (assume layer norm)
        model = DictLinearAE(rep_dim, hid_dim, cfg.get("norm", "ln"))
        model.load_state_dict(state_dict)
        model.eval()

        # Store config info as attributes for easy access
        model.model_name = model_name
        model.layer = layer
        model.rep_dim = rep_dim

        return model

    finally:
        # Clean up temporary directory if it was created
        if temp_dir is not None:
            shutil.rmtree(temp_dir, ignore_errors=True)


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
    from transformers import (
        GPTNeoXForCausalLM,
        AutoTokenizer,
        AutoModelForCausalLM,
    )

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
            "google/gemma-2-2b-it",
            cache_dir="./gemma-2-2b-it",
        ).to(device)
        tokenizer = AutoTokenizer.from_pretrained(
            "google/gemma-2-2b-it",
            cache_dir="./gemma-2-2b-it",
        )
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

    # Get model device
    model_device = next(model.parameters()).device

    with t.no_grad():
        for i in range(0, len(embeddings), batch_size):
            batch = embeddings[i : i + batch_size].to(model_device)
            _, activations = model(batch)  # Get hidden activations
            all_activations.append(activations.cpu())

    return t.cat(all_activations, dim=0)


def heuristic_feature_ranking_binary(X, y, method="lr"):
    """Rank features using heuristic methods for sparse probing."""
    if method == "max_mean_diff":
        pos_class = y == 1
        neg_class = y == 0
        mean_diff = abs(
            X[pos_class, :].mean(axis=0) - X[neg_class, :].mean(axis=0)
        )
        sorted_idxs = np.argsort(mean_diff)
    elif method == "lr":
        lr = LogisticRegression(
            class_weight="balanced", penalty="l1", solver="saga", C=0.1
        )
        lr.fit(X, y)
        sorted_idxs = np.argsort(np.abs(lr.coef_[0]))
    else:
        raise ValueError(f"Unknown sparse method: {method}")
    return sorted_idxs


def train_probes(
    train_activations,
    train_labels,
    test_activations,
    test_labels,
    probe_type="binary",
    seed=42,
    sparse="lr",
    k=10,
):
    """
    Train logistic regression probes with optional sparse feature selection.

    This unified function handles both binary and multiclass probing, with optional
    top-k feature selection based on heuristic methods.

    Args:
        train_activations: Training activations (N x D)
        train_labels: Training labels (N,)
        test_activations: Test activations (M x D)
        test_labels: Test labels (M,)
        probe_type: Either "binary" or "multiclass"
        seed: Random seed
        sparse: Sparse probing method ("max_mean_diff" or "lr"), None for standard
        k: Number of top features to use when sparse is not None

    Returns:
        If sparse is None: trained classifier
        If sparse is not None: tuple of (classifier, top_neurons)
    """
    if probe_type not in ["binary", "multiclass"]:
        raise ValueError(
            f"Invalid probe_type: {probe_type}. Must be 'binary' or 'multiclass'"
        )

    # Determine solver based on probe type
    if probe_type == "binary":
        solver = "newton-cholesky"
        multi_class = "auto"
    else:  # multiclass
        solver = "lbfgs"
        multi_class = "multinomial"

    # Handle sparse feature selection
    if sparse is not None:
        # Select top-k features using heuristic ranking
        sorted_neurons = heuristic_feature_ranking_binary(
            train_activations, train_labels, method=sparse
        )
        top_neurons = sorted_neurons[-k:]

        # Train on selected features
        X_train_sub = train_activations[:, top_neurons]
        X_test_sub = test_activations[:, top_neurons]

        classifier = LogisticRegression(
            random_state=seed,
            max_iter=1000,
            class_weight="balanced",
            solver=solver,
            multi_class=multi_class,
        )
        classifier.fit(X_train_sub, train_labels)

        # Evaluate
        train_acc = classifier.score(X_train_sub, train_labels)
        test_acc = classifier.score(X_test_sub, test_labels)

        print(f"Train Accuracy: {train_acc:.2f}")
        print(f"Test Accuracy: {test_acc:.2f}")

        return classifier, top_neurons
    else:
        # Standard probing: use all features
        classifier = LogisticRegression(
            random_state=seed,
            max_iter=1000,
            class_weight="balanced",
            solver=solver,
            multi_class=multi_class,
        )
        classifier.fit(train_activations, train_labels)

        # Evaluate
        train_accuracy = classifier.score(train_activations, train_labels)
        test_accuracy = classifier.score(test_activations, test_labels)

        print(f"Train Accuracy: {train_accuracy:.2f}")
        print(f"Test Accuracy: {test_accuracy:.2f}")

        return classifier


def get_features_and_values_from_probes(output_dir):
    """Auto-discover available probes from directory."""
    concepts = {}
    files = os.listdir(output_dir)
    for file in files:
        if not file.endswith(".joblib"):
            continue
        basename, ext = os.path.splitext(file)
        splits = basename.split("_")

        # Handle different filename formats
        if len(splits) >= 2:
            key = splits[0]
            value = splits[1]
            # Remove layer suffix if present
            if value.startswith("layer"):
                continue
            if key not in concepts:
                concepts[key] = set()
            concepts[key].add(value)
    return concepts


def compute_correlation_matrix(acts, labels):
    """
    Compute Pearson correlation matrix between activations and labels.

    This is the canonical correlation computation used throughout the codebase.
    It computes correlations between each feature (column of acts) and each label.

    Args:
        acts: Activations tensor (N x F) where N=samples, F=features
        labels: Either a dict of label vectors {label_name: [N]} or a tensor (N x L)
                where L=number of labels

    Returns:
        corr_matrix: Correlation matrix (F x L) where F=features, L=labels
        label_names: List of label names (if labels was a dict) or None
    """
    # Handle dict input
    if isinstance(labels, dict):
        label_names = list(labels.keys())
        label_matrix = t.stack(
            [t.Tensor(labels[l]) for l in label_names], dim=0
        )  # L x N
    else:
        # Assume labels is already a tensor (N x L) or (L x N)
        if labels.ndim == 1:
            labels = labels.unsqueeze(1)  # Make it N x 1
        # Check if we need to transpose
        if labels.shape[0] == acts.shape[0]:  # N x L format
            label_matrix = labels.T  # Convert to L x N
        else:
            label_matrix = labels  # Already L x N
        label_names = None

    # Center activations
    acts_centered = acts - acts.mean(dim=0, keepdim=True)
    acts_std = acts_centered.norm(dim=0, keepdim=True)

    # Center labels (convert to float if boolean)
    label_matrix_float = label_matrix.T.float()  # N x L
    label_matrix_centered = label_matrix_float - label_matrix_float.mean(
        dim=0, keepdim=True
    )
    label_matrix_std = label_matrix_centered.norm(dim=0, keepdim=True)

    # Compute correlation: (F x N) @ (N x L) = F x L
    numerator = acts_centered.T @ label_matrix_centered
    denominator = acts_std.T * label_matrix_std  # Broadcasting

    # Prevent NaNs from division by zero
    mask = denominator != 0
    corr_matrix = t.zeros_like(numerator)
    corr_matrix[mask] = numerator[mask] / denominator[mask]

    return corr_matrix, label_names


def evaluate_probe(probe, test_activations, test_labels):
    """Simple probe evaluation without intervention."""
    test_accuracy = probe.score(test_activations, test_labels)
    return test_accuracy


def extract_activations_with_intervention(
    x_list, f_list_orig, dictionary, feature_idx, model_path=None
):
    """Apply causal intervention by amplifying a specific SAE feature."""
    x_int_list = []
    feature_max = f_list_orig[:, feature_idx].max().item()

    for idx, x in enumerate(x_list):
        x = t.tensor(x, dtype=t.float32).to("cuda")

        # Get original reconstruction and features
        x_hat_org, f_orig = dictionary(x, return_hidden=True)

        # Create intervention: amplify target feature by 5×
        f_new = f_orig.clone()
        f_new[:, feature_idx] = feature_max * 5.0

        # Decode modified features
        x_hat_int = dictionary.decode(f_new)

        # Compute delta and apply to original
        x_hat_delta = x_hat_int - x_hat_org
        x_int = x + x_hat_delta
        x_int_list.append(x_int.squeeze())

    return t.stack(x_int_list, dim=0)


def extract_activations_with_multi_intervention(
    x_list, f_list_orig, dictionary, top_feature_idxs, steer_list, coef=5.0
):
    """Multi-concept steering - intervene on multiple features simultaneously."""
    x_int_list = []
    feature_maxes = {
        c: f_list_orig[:, top_feature_idxs[c]].max().item() for c in steer_list
    }

    # Coefficient distribution logic
    if len(steer_list) <= 1:
        per_feature_coef = coef
    elif len(steer_list) == 2:
        per_feature_coef = coef
    else:
        raise NotImplementedError(
            "Coefficient distribution for > 2 concepts not yet supported."
        )

    for idx, x in enumerate(x_list):
        x = t.tensor(x, dtype=t.float32).to("cuda")

        # Get original reconstruction and features
        x_hat_org, f_orig = dictionary(x, return_hidden=True)

        # Apply multi-concept intervention
        f_new = f_orig.clone()
        for concept in steer_list:
            f_new[:, top_feature_idxs[concept]] = (
                per_feature_coef * feature_maxes[concept]
            )

        # Decode and compute delta
        x_hat_int = dictionary.decode(f_new)
        x_hat_delta = x_hat_int - x_hat_org
        x_int = x + x_hat_delta
        x_int_list.append(x_int.squeeze())

    return t.stack(x_int_list, dim=0)


def test_all_with_interventions(
    model,
    dictionary,
    train_activations,
    test_activations_dict,
    test_labels_dict,
    top_features,
    sae_acts,
    output_dir,
    verbose=False,
):
    """
    Test all concept interventions on all probes.

    Args:
        model: Language model
        dictionary: SAE dictionary
        train_activations: Training activations for feature selection
        test_activations_dict: Dict of {concept_str: test_activations}
        test_labels_dict: Dict of {concept_str: test_labels}
        top_features: Dict of {concept_str: feature_idx}
        sae_acts: SAE activations for intervention
        output_dir: Directory with saved probes
        verbose: Print progress

    Returns:
        score_matrix: Matrix of intervention effects
    """
    # Get available concepts from probes
    features = get_features_and_values_from_probes(output_dir)

    # Build index mapping
    IDX = {}
    score_matrix = []
    num_concepts = 0

    for train_concept_key, values in features.items():
        for train_concept_value in values:
            idx = len(score_matrix)
            train_concept_str = f"{train_concept_key}-{train_concept_value}"
            IDX[train_concept_str] = idx
            score_matrix.append([])
            num_concepts += 1

    for i in range(len(score_matrix)):
        score_matrix[i] = [0.0] * num_concepts

    # Test interventions
    for train_concept_key, train_values in features.items():
        for train_concept_value in train_values:
            train_concept_str = f"{train_concept_key}-{train_concept_value}"

            if train_concept_str not in top_features:
                if verbose:
                    print(f"No top feature for {train_concept_str}, skipping")
                continue

            top_feature = top_features[train_concept_str]

            for test_concept_key, test_values in features.items():
                for test_concept_value in test_values:
                    test_concept_str = (
                        f"{test_concept_key}-{test_concept_value}"
                    )

                    if test_concept_str not in test_activations_dict:
                        continue

                    if verbose:
                        print(f"{train_concept_str} -> {test_concept_str}")

                    # Load probe
                    probe_filename = (
                        f"{test_concept_key}_{test_concept_value}.joblib"
                    )
                    probe_path = os.path.join(output_dir, probe_filename)

                    if not os.path.exists(probe_path):
                        if verbose:
                            print(f"Probe not found: {probe_path}")
                        continue

                    probe = joblib.load(probe_path)

                    # Get test activations
                    test_activations_org = test_activations_dict[
                        test_concept_str
                    ]
                    test_labels = test_labels_dict[test_concept_str]

                    # Apply intervention
                    with t.no_grad():
                        test_activations_int = (
                            extract_activations_with_intervention(
                                test_activations_org,
                                sae_acts,
                                dictionary,
                                top_feature,
                            )
                        )

                    # Compute probe response difference
                    scores_org = probe.decision_function(test_activations_org)
                    scores_int = probe.decision_function(
                        test_activations_int.detach().cpu().numpy()
                    )
                    scores_delta = scores_int - scores_org
                    mean_delta = scores_delta.mean()

                    if verbose:
                        print(f"  Delta: {mean_delta:.4f}")

                    score_matrix[IDX[train_concept_str]][
                        IDX[test_concept_str]
                    ] = mean_delta

    return score_matrix, IDX


def get_attributions_w_hooks(
    model,
    tokenizer,
    submodule_steer_name,
    submodule_probe_name,
    dictionary,
    probe,
    batch,
    logit_idx=None,
    use_sparsemax=False,
    scaler=None,
):
    """
    Compute feature attributions using hooks instead of nnsight's model.trace().

    This implements the main attribution logic:
    1. Replace activations with SSAE outputs (via hook on submodule_steer)
    2. Compute probe logits (capturing activations at submodule_probe)
    3. Backpropagate from probe logits
    4. Multiply feature gradients by feature activations (gradient * activation attribution)

    Args:
        model: Language model (GPTNeoXForCausalLM or AutoModelForCausalLM)
        tokenizer: Tokenizer for the model
        submodule_steer_name: String name of the submodule to intercept for SAE (e.g., 'gpt_neox.layers.5')
        submodule_probe_name: String name of the submodule to capture activations for probe (e.g., 'gpt_neox.layers.5')
        dictionary: SSAE dictionary model
        probe: Trained probe (sklearn LogisticRegression or similar)
        batch: List of text strings to process
        logit_idx: Index of logit to use (currently unused, kept for API compatibility)
        use_sparsemax: Whether the SAE uses sparsemax/MP activation (affects reshaping)

    Returns:
        Tuple of (f_grad_att, x_grad_att, cache, logits):
            - f_grad_att: Feature gradient attributions (gradient * activation)
            - x_grad_att: Residual stream gradient attributions
            - cache: Dict containing intermediate activations
            - logits: Probe logits
    """
    cache = {"x": None, "f": None, "resid_stream": None, "f_flat": None}

    # Get submodules by name (handle ModuleList indexing for Gemma)
    submodule_steer = get_submodule_with_index(model, submodule_steer_name)
    submodule_probe = get_submodule_with_index(model, submodule_probe_name)

    def _set_sae_activation(_module, _input, _output):
        """Hook to replace activations with SAE reconstruction."""
        # Handle tuple outputs
        if isinstance(_output, tuple):
            resid_stream = _output[0]
        else:
            resid_stream = _output

        resid_stream.requires_grad_(True)
        resid_stream.retain_grad()

        # Process through SAE
        if use_sparsemax:
            # Flatten for sparsemax/MP models
            x_hat_flat, f_flat = dictionary(
                resid_stream.flatten(start_dim=0, end_dim=1)
            )
            x_hat = x_hat_flat.view(
                resid_stream.shape[0], resid_stream.shape[1], -1
            )
            f = f_flat.view(resid_stream.shape[0], resid_stream.shape[1], -1)
            f_flat.requires_grad_(True)
            f_flat.retain_grad()
            cache["f_flat"] = f_flat
        else:
            x_hat, f = dictionary(resid_stream)

        if not f.requires_grad:
            f.requires_grad_(True)
        f.retain_grad()

        # Create residual that doesn't receive gradients
        # This ensures gradients only flow through SAE reconstruction
        resid = (resid_stream - x_hat).detach()
        resid.grad = t.zeros_like(resid)
        x_recon = x_hat + resid

        cache["f"] = f
        cache["resid_stream"] = resid_stream

        # Replace output with reconstruction
        if isinstance(_output, tuple):
            _output = (_output[0].clone(),) + _output[1:]
            _output[0][:] = x_recon
        else:
            _output = x_recon

        return _output

    def _get_activations(_module, _input, _output):
        """Hook to capture activations for the probe."""
        if isinstance(_output, tuple):
            assert len(_output) == 1
            _output = _output[0]
        cache["x"] = _output
        return None

    # Register hooks
    hook_steer = submodule_steer.register_forward_hook(_set_sae_activation)
    hook_probe = submodule_probe.register_forward_hook(_get_activations)

    # Enable gradients and run forward pass
    t.set_grad_enabled(True)
    batch_tokenized = tokenizer(batch, return_tensors="pt", padding=True).to(
        model.device
    )

    _ = model(**batch_tokenized)

    # Remove hooks
    hook_steer.remove()
    hook_probe.remove()

    # Process probe activations
    submod_acts = cache["x"].sum(dim=1).squeeze(dim=-1)

    # Normalize if scaler is provided (must match training normalization)
    # Use differentiable PyTorch operations instead of sklearn transform
    if scaler is not None:
        # Extract mean and scale from fitted scaler
        # Use same dtype as model activations
        scaler_mean = t.tensor(scaler.mean_, dtype=submod_acts.dtype).to(
            model.device
        )
        scaler_scale = t.tensor(scaler.scale_, dtype=submod_acts.dtype).to(
            model.device
        )
        # Apply standardization: (x - mean) / scale
        submod_acts = (submod_acts - scaler_mean) / scaler_scale

    # Apply probe weights in PyTorch to maintain gradient flow
    # probe is a sklearn LogisticRegression, extract weights
    # For binary (2 classes): coef_ shape is (1, n_features)
    #   - Class 0 logit: -coef_[0] - intercept_[0]
    #   - Class 1 logit: +coef_[0] + intercept_[0]
    # For multiclass (>2 classes): coef_ shape is (n_classes, n_features)
    #   - Class i logit: coef_[i] + intercept_[i]
    # Use same dtype as model activations
    if logit_idx is not None:
        # Check if binary (2 classes) or true multiclass (>2 classes)
        if probe.coef_.shape[0] == 1:
            # Binary classification with 2 classes
            # Class 0: use negative weights, Class 1: use positive weights
            sign = -1.0 if logit_idx == 0 else 1.0
            probe_weights = t.tensor(
                sign * probe.coef_[0], dtype=submod_acts.dtype
            ).to(model.device)
            probe_bias = t.tensor(
                sign * probe.intercept_[0], dtype=submod_acts.dtype
            ).to(model.device)
        else:
            # True multiclass: select specific class weights
            probe_weights = t.tensor(
                probe.coef_[logit_idx], dtype=submod_acts.dtype
            ).to(model.device)
            probe_bias = t.tensor(
                probe.intercept_[logit_idx], dtype=submod_acts.dtype
            ).to(model.device)
    else:
        # No logit_idx specified: binary case, use positive direction
        probe_weights = t.tensor(
            probe.coef_.squeeze(), dtype=submod_acts.dtype
        ).to(model.device)
        probe_bias = t.tensor(probe.intercept_[0], dtype=submod_acts.dtype).to(
            model.device
        )

    # Compute logits maintaining gradients
    # submod_acts: (batch, n_features), probe_weights: (n_features,)
    logits = submod_acts @ probe_weights + probe_bias

    # Backpropagate from probe logits
    metric = t.sum(logits)
    metric.backward()

    # Extract gradients
    if use_sparsemax:
        f_grad = cache["f_flat"].grad.detach()
        f_grad = f_grad.reshape(
            cache["resid_stream"].shape[0], cache["resid_stream"].shape[1], -1
        )
    else:
        f_grad = cache["f"].grad.detach()

    x_grad = cache["resid_stream"].grad.detach()

    # Compute gradient * activation attributions
    f_grad_att = f_grad * cache["f"].detach()
    x_grad_att = x_grad * cache["resid_stream"].detach()

    return f_grad_att, x_grad_att, cache, logits


def get_pair_attributions_w_hooks(
    model,
    tokenizer,
    submodule_steer_name,
    submodule_probe_name,
    dictionary,
    probe,
    batch_1,
    batch_2,
    use_sparsemax=False,
    scaler=None,
    logit_idx=None,
):
    """
    Compute feature attributions for difference between two batches using hooks.

    This function computes attributions for how SAE features explain the difference
    in probe predictions between two batches (x1 and x2). Unlike get_attributions_w_hooks
    which processes a single batch, this function:
    1. Runs x1 and caches activations (no gradients)
    2. Runs x2 and computes delta_x = x2 - x1
    3. Passes delta_x through SSAE to get delta_x_hat and delta_f
    4. Reconstructs x2 as x1 + delta_x_hat + (delta_x - delta_x_hat).detach()
    5. Computes probe predictions for both x1 and x2
    6. Backpropagates from change in probe predictions: logits2 - logits1
    7. Returns delta_f * grad(delta_f) attributions

    This is specifically designed for SAEs trained on differences between embeddings.

    Args:
        model: Language model (GPTNeoXForCausalLM or AutoModelForCausalLM)
        tokenizer: Tokenizer for the model
        submodule_steer_name: String name of the submodule to intercept for SAE (e.g., 'gpt_neox.layers.5')
        submodule_probe_name: String name of the submodule to capture activations for probe
        dictionary: SSAE dictionary model trained on embedding differences
        probe: Trained probe (sklearn LogisticRegression)
        batch_1: List of text strings (x1 examples)
        batch_2: List of text strings (x2 examples, must be same length as batch_1)
        use_sparsemax: Whether the SAE uses sparsemax/MP activation
        scaler: StandardScaler for normalizing activations (optional)
        logit_idx: For multiclass probes, which class logit to compute gradients for (optional)

    Returns:
        Tuple of (delta_f_att, cache, (logits1, logits2)):
            - delta_f_att: Feature gradient attributions for the difference (gradient * delta_f)
            - cache: Dict containing intermediate activations and deltas
            - (logits1, logits2): Tuple of probe logits for batch_1 and batch_2
    """
    # Validate inputs
    assert len(batch_1) == len(
        batch_2
    ), f"batch_1 and batch_2 must have same length, got {len(batch_1)} and {len(batch_2)}"

    cache = {
        "resid_stream_1": None,
        "resid_stream_2": None,
        "delta_f": None,
        "delta_x": None,
        "x1_probe": None,
        "x2_probe": None,
        "delta_f_flat": None,
    }

    submodule_steer = get_submodule_with_index(model, submodule_steer_name)
    submodule_probe = get_submodule_with_index(model, submodule_probe_name)

    # ---- 1. Run x1 and cache resid_stream_1 ----
    def hook_steer_x1(_module, _input, _output):
        """Hook to capture x1 activations without gradients."""
        if isinstance(_output, tuple):
            resid = _output[0]
        else:
            resid = _output
        cache["resid_stream_1"] = resid.detach()
        return None

    def hook_probe_x1(_module, _input, _output):
        """Hook to capture probe activations for x1."""
        if isinstance(_output, tuple):
            _output = _output[0]
        cache["x1_probe"] = _output.detach()
        return None

    h_s1 = submodule_steer.register_forward_hook(hook_steer_x1)
    h_p1 = submodule_probe.register_forward_hook(hook_probe_x1)

    with t.no_grad():
        batch1_tok = tokenizer(batch_1, return_tensors="pt", padding=True).to(
            model.device
        )
        _ = model(**batch1_tok)

    h_s1.remove()
    h_p1.remove()

    # ---- 2. Run x2 with SSAE difference-based steering hook ----
    def hook_steer_x2(_module, _input, _output):
        """Hook to replace activations with SSAE reconstruction of difference."""
        if isinstance(_output, tuple):
            resid2 = _output[0]
        else:
            resid2 = _output

        # Ensure gradient tracking on x2
        if not resid2.requires_grad:
            resid2.requires_grad_(True)
        resid2.retain_grad()

        resid1 = cache["resid_stream_1"]  # (batch, seq, d) - no gradients

        # Compute delta_x: x2 - x1
        delta_x = resid2 - resid1
        cache["delta_x"] = delta_x

        # Pass delta_x through SSAE to get reconstruction and features
        if use_sparsemax:
            # Flatten for sparsemax/MP models
            delta_x_flat = delta_x.flatten(start_dim=0, end_dim=1)
            delta_x_hat_flat, delta_f_flat = dictionary(delta_x_flat)
            delta_x_hat = delta_x_hat_flat.view_as(delta_x)
            delta_f = delta_f_flat.view(delta_x.shape[0], delta_x.shape[1], -1)
            delta_f_flat.requires_grad_(True)
            delta_f_flat.retain_grad()
            cache["delta_f_flat"] = delta_f_flat
        else:
            delta_x_hat, delta_f = dictionary(delta_x)

        if not delta_f.requires_grad:
            delta_f.requires_grad_(True)
        delta_f.retain_grad()

        # Detach unexplained part of delta_x (gradients only through SSAE reconstruction)
        delta_resid = (delta_x - delta_x_hat).detach()

        # Reconstruct x2 from: x1 + explained_delta + unexplained_delta
        delta_x_recon = delta_x_hat + delta_resid
        resid2_recon = resid1 + delta_x_recon

        cache["delta_f"] = delta_f
        cache["resid_stream_2"] = resid2

        # Replace output with reconstruction
        if isinstance(_output, tuple):
            return (resid2_recon,) + _output[1:]
        else:
            return resid2_recon

    def hook_probe_x2(_module, _input, _output):
        """Hook to capture probe activations for x2."""
        if isinstance(_output, tuple):
            _output = _output[0]
        cache["x2_probe"] = _output
        return None

    h_s2 = submodule_steer.register_forward_hook(hook_steer_x2)
    h_p2 = submodule_probe.register_forward_hook(hook_probe_x2)

    with t.enable_grad():
        batch2_tok = tokenizer(batch_2, return_tensors="pt", padding=True).to(
            model.device
        )
        _ = model(**batch2_tok)

    h_s2.remove()
    h_p2.remove()

    # ---- 3. Build probe metric on change in representation ----
    x1_probe = cache["x1_probe"]  # (batch, seq, d_probe)
    x2_probe = cache["x2_probe"]  # (batch, seq, d_probe)

    # Sum over sequence dimension
    submod_acts_1 = x1_probe.sum(dim=1).squeeze(dim=-1).detach()
    submod_acts_2 = x2_probe.sum(dim=1).squeeze(dim=-1)

    # Optional scaler normalization (must match training normalization)
    if scaler is not None:
        mean = t.tensor(
            scaler.mean_, dtype=submod_acts_2.dtype, device=model.device
        )
        scale = t.tensor(
            scaler.scale_, dtype=submod_acts_2.dtype, device=model.device
        )
        submod_acts_1 = (submod_acts_1 - mean) / scale
        submod_acts_2 = (submod_acts_2 - mean) / scale

    # Apply probe weights (same logic as get_attributions_w_hooks)
    # probe is a sklearn LogisticRegression
    # For binary (2 classes): coef_ shape is (1, n_features)
    # For multiclass (>2 classes): coef_ shape is (n_classes, n_features)
    if logit_idx is not None:
        # Check if binary (2 classes) or true multiclass (>2 classes)
        if probe.coef_.shape[0] == 1:
            # Binary classification with 2 classes
            # Class 0: use negative weights, Class 1: use positive weights
            sign = -1.0 if logit_idx == 0 else 1.0
            probe_weights = t.tensor(
                sign * probe.coef_[0], dtype=submod_acts_2.dtype
            ).to(model.device)
            probe_bias = t.tensor(
                sign * probe.intercept_[0], dtype=submod_acts_2.dtype
            ).to(model.device)
        else:
            # True multiclass: select specific class weights
            probe_weights = t.tensor(
                probe.coef_[logit_idx], dtype=submod_acts_2.dtype
            ).to(model.device)
            probe_bias = t.tensor(
                probe.intercept_[logit_idx], dtype=submod_acts_2.dtype
            ).to(model.device)
    else:
        # No logit_idx specified: binary case, use positive direction
        probe_weights = t.tensor(
            probe.coef_.squeeze(), dtype=submod_acts_2.dtype
        ).to(model.device)
        probe_bias = t.tensor(
            probe.intercept_[0], dtype=submod_acts_2.dtype
        ).to(model.device)

    # Compute logits for both batches
    logits1 = submod_acts_1 @ probe_weights + probe_bias
    logits2 = submod_acts_2 @ probe_weights + probe_bias

    # Metric: change in probe prediction (only logits2 has gradients)
    metric = t.sum(logits2 - logits1.detach())
    metric.backward()

    # ---- 4. Extract gradient × delta_f attribution ----
    if use_sparsemax:
        delta_f_grad = cache["delta_f_flat"].grad
        if delta_f_grad is None:
            raise RuntimeError(
                "No gradients computed for delta_f_flat. Check backward pass."
            )
        delta_f_grad = delta_f_grad.detach()
        delta_f_grad = delta_f_grad.reshape(
            cache["resid_stream_2"].shape[0],
            cache["resid_stream_2"].shape[1],
            -1,
        )
    else:
        delta_f_grad = cache["delta_f"].grad
        if delta_f_grad is None:
            raise RuntimeError(
                "No gradients computed for delta_f. Check backward pass."
            )
        delta_f_grad = delta_f_grad.detach()

    # Attribution: element-wise product of delta_f and its gradient
    delta_f_att = cache["delta_f"].detach() * delta_f_grad

    return delta_f_att, cache, (logits1.detach(), logits2.detach())


def find_top_k_features_by_attribution(
    model,
    tokenizer,
    submodule_steer_name,
    submodule_probe_name,
    dictionary,
    probe,
    sentences,
    k=10,
    use_sparsemax=False,
    batch_size=32,
    scaler=None,
    logit_idx=None,
    use_abs=True,
):
    """
    Find top k SAE features using gradient attribution method.

    This function computes gradient-based feature attributions across a dataset
    and selects the k most important features. The attribution pipeline:
    1. For each batch: compute f_grad_att via get_attributions_w_hooks
    2. Sum attributions over sequence dimension (total contribution per sentence)
    3. Accumulate across all batches
    4. Sum across all examples and select top k by magnitude

    Args:
        model: Language model (GPTNeoXForCausalLM or AutoModelForCausalLM)
        tokenizer: Tokenizer for the model
        submodule_steer_name: String name of submodule to intercept for SAE
        submodule_probe_name: String name of submodule to capture for probe
        dictionary: SSAE dictionary model
        probe: Trained probe (sklearn LogisticRegression)
        sentences: List of text strings to process
        k: Number of top features to return
        use_sparsemax: Whether SAE uses sparsemax/MP activation
        batch_size: Number of sentences to process per batch
        scaler: StandardScaler for normalizing activations (optional)
        logit_idx: For multiclass probes, which class logit to compute gradients for (optional)

    Returns:
        Tuple of (top_k_scores, top_k_indices, all_scores):
            - top_k_scores: Attribution scores for top k features (shape: k)
            - top_k_indices: Indices of top k features (shape: k)
            - all_scores: Full attribution scores for all features (shape: num_features)
    """
    # Get number of features from dictionary
    num_features = dictionary.encoder.weight.shape[0]

    # Initialize accumulator for attributions across all examples
    acts_all = t.zeros(len(sentences), num_features)

    # Process in batches
    num_batches = (len(sentences) + batch_size - 1) // batch_size

    for idx in tqdm(
        range(num_batches), desc="Computing gradient attributions"
    ):
        start_idx = idx * batch_size
        end_idx = min(start_idx + batch_size, len(sentences))
        batch = sentences[start_idx:end_idx]

        # Compute gradient attributions for this batch
        f_grad_att, x_grad_att, cache, logits = get_attributions_w_hooks(
            model=model,
            tokenizer=tokenizer,
            submodule_steer_name=submodule_steer_name,
            submodule_probe_name=submodule_probe_name,
            dictionary=dictionary,
            probe=probe,
            batch=batch,
            logit_idx=logit_idx,  # Pass through for multiclass support
            use_sparsemax=use_sparsemax,
            scaler=scaler,
        )

        # Sum over sequence dimension: (batch, seq_len, features) -> (batch, features)
        f_grad_att_summed = f_grad_att.sum(dim=1)

        # Store in accumulator
        len_batch = len(batch)
        acts_all[start_idx : start_idx + len_batch] = (
            f_grad_att_summed.detach().cpu()
        )

    # Aggregate across all examples: (num_examples, features) -> (features,)
    all_scores = acts_all.sum(dim=0)

    # Select top k features by ABSOLUTE magnitude (not just largest)
    # This is important because gradient attribution can be negative
    abs_scores = all_scores.abs()
    _, top_k_indices = t.topk(abs_scores, k=k)

    # Get the actual scores (with sign preserved) for the selected indices
    top_k_scores = all_scores[top_k_indices]

    return top_k_scores, top_k_indices, all_scores


def find_top_k_features_by_attribution_pairs(
    model,
    tokenizer,
    submodule_steer_name,
    submodule_probe_name,
    dictionary,
    probe,
    sentences_1,
    sentences_2,
    k=10,
    use_sparsemax=False,
    batch_size=32,
    scaler=None,
    logit_idx=None,
):
    num_features = dictionary.encoder.weight.shape[0]
    acts_all = t.zeros(len(sentences_1), num_features)

    num_batches = (len(sentences_1) + batch_size - 1) // batch_size

    for idx in range(num_batches):
        start = idx * batch_size
        end = min(start + batch_size, len(sentences_1))
        batch1 = sentences_1[start:end]
        batch2 = sentences_2[start:end]

        delta_f_att, cache, (logits1, logits2) = get_pair_attributions_w_hooks(
            model=model,
            tokenizer=tokenizer,
            submodule_steer_name=submodule_steer_name,
            submodule_probe_name=submodule_probe_name,
            dictionary=dictionary,
            probe=probe,
            batch_1=batch1,
            batch_2=batch2,
            use_sparsemax=use_sparsemax,
            scaler=scaler,
            logit_idx=logit_idx,
        )

        # delta_f_att: (batch, seq_len, num_features)
        delta_f_att_summed = delta_f_att.sum(dim=1)  # (batch, num_features)
        acts_all[start : start + len(batch1)] = (
            delta_f_att_summed.detach().cpu()
        )

    all_scores = acts_all.sum(dim=0)  # (num_features,)
    abs_scores = all_scores.abs()
    _, top_k_indices = t.topk(abs_scores, k=k)
    top_k_scores = all_scores[top_k_indices]
    return top_k_scores, top_k_indices, all_scores


def train_sparse_probe_on_top_k(
    sae_activations_train,
    sae_activations_test,
    labels_train,
    labels_test,
    top_k_indices,
    seed=42,
    verbose=False,
):
    """
    Train a sparse probe on pre-selected top-k features and compute MCC.

    Simple function that:
    1. Takes pre-selected top-k feature indices
    2. Trains a probe on those features
    3. Computes MCC (correlation between probe logits and labels)

    Args:
        sae_activations_train: Training SAE activations (N_train x F)
        sae_activations_test: Test SAE activations (N_test x F)
        labels_train: Binary training labels (N_train,)
        labels_test: Binary test labels (N_test,)
        top_k_indices: Indices of top-k features to use (array-like, length k)
        seed: Random seed
        verbose: Print progress

    Returns:
        results: Dict containing:
            - "probe": Trained sparse classifier
            - "train_acc": Training accuracy
            - "test_acc": Test accuracy
            - "mcc": Absolute Pearson correlation between probe logits and test labels
            - "correlation": Raw correlation coefficient
            - "p_value": P-value from correlation test
    """
    # Convert to numpy if needed
    if isinstance(sae_activations_train, t.Tensor):
        sae_activations_train = sae_activations_train.numpy()
    if isinstance(sae_activations_test, t.Tensor):
        sae_activations_test = sae_activations_test.numpy()
    if isinstance(labels_train, t.Tensor):
        labels_train = labels_train.numpy()
    if isinstance(labels_test, t.Tensor):
        labels_test = labels_test.numpy()
    if isinstance(top_k_indices, t.Tensor):
        top_k_indices = top_k_indices.numpy()

    # Select top-k features
    X_train_sparse = sae_activations_train[:, top_k_indices]
    X_test_sparse = sae_activations_test[:, top_k_indices]

    # Train sparse probe
    classifier = LogisticRegression(
        random_state=seed,
        max_iter=1000,
        class_weight="balanced",
        solver="newton-cholesky",
    )
    classifier.fit(X_train_sparse, labels_train)

    # Evaluate
    train_acc = classifier.score(X_train_sparse, labels_train)
    test_acc = classifier.score(X_test_sparse, labels_test)

    # Compute MCC
    logits = classifier.decision_function(X_test_sparse)
    correlation, p_value = pearsonr(logits, labels_test)
    mcc = abs(correlation)

    if verbose:
        print(
            f"Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}, MCC: {mcc:.4f}"
        )

    return {
        "probe": classifier,
        "train_acc": train_acc,
        "test_acc": test_acc,
        "mcc": mcc,
        "correlation": correlation,
        "p_value": p_value,
    }


def sweep_k_values_for_plots(
    sae_activations_train,
    sae_activations_test,
    labels_train,
    labels_test,
    all_feature_scores,
    k_values,
    seed=42,
    verbose=True,
    compute_activation_mcc=False,
    mcc_union_features=False,
):
    """
    Sweep over k values to create MCC vs k plots (single concept version).

    Supports both single concept and multi-concept workflows:
    - Single concept: Pass labels as arrays, all_feature_scores as tensor
    - Multi concept: Pass labels as dicts, all_feature_scores as dict

    Usage example (single concept):
        # Get feature scores from trained probe coefficients
        all_scores = np.abs(probe.coef_[0])
        results = sweep_k_values_for_plots(
            sae_train, sae_test, labels_train, labels_test,
            all_scores, k_values=[1, 3, 5, 10, 25, 50]
        )
        plt.plot(results["k_values"], results["mcc_values"])

    Usage example (multi concept):
        results = sweep_k_values_for_plots(
            sae_train, sae_test,
            {"domain-science": labels_train_sci, "sentiment-positive": labels_train_sent},
            {"domain-science": labels_test_sci, "sentiment-positive": labels_test_sent},
            {"domain-science": all_scores_sci, "sentiment-positive": all_scores_sent},
            k_values=[1, 3, 5, 10, 25, 50],
            compute_activation_mcc=True
        )

    Args:
        sae_activations_train: Training SAE activations (N_train x F)
        sae_activations_test: Test SAE activations (N_test x F)
        labels_train: Binary labels (N_train,) OR dict of {concept: labels_train}
        labels_test: Binary labels (N_test,) OR dict of {concept: labels_test}
        all_feature_scores: LR feature scores (F,) OR dict of {concept: scores}
                        Should be np.abs(probe.coef_[0]) from trained probes
        k_values: List of k values to test
        seed: Random seed
        verbose: Print progress
        compute_activation_mcc: If True, also compute activation-label MCC using
                            compute_correlation_matrix (needed for right plot)
        mcc_union_features: If True, use union of top-k features across concepts for MCC.
                        If False (default), compute MCC per concept and average.

    Returns:
        If single concept:
            Dict with "k_values", "mcc_values", "train_acc_values", etc.
        If multi concept:
            Dict with "k_values" and per-concept results, plus "aggregate_mcc"
    """
    # Detect if multi-concept (dict input)
    is_multi_concept = isinstance(labels_train, dict)

    if is_multi_concept:
        return _sweep_k_multi_concept(
            sae_activations_train,
            sae_activations_test,
            labels_train,
            labels_test,
            all_feature_scores,
            k_values,
            seed,
            verbose,
            compute_activation_mcc,
            mcc_union_features,
        )
    else:
        return _sweep_k_single_concept(
            sae_activations_train,
            sae_activations_test,
            labels_train,
            labels_test,
            all_feature_scores,
            k_values,
            seed,
            verbose,
        )


def _sweep_k_single_concept(
    sae_activations_train,
    sae_activations_test,
    labels_train,
    labels_test,
    all_feature_scores,
    k_values,
    seed,
    verbose,
):
    """Helper for single concept k-sweep."""
    mcc_values = []
    train_acc_values = []
    test_acc_values = []
    correlation_values = []

    if isinstance(all_feature_scores, np.ndarray):
        all_feature_scores = t.tensor(all_feature_scores)

    for k in tqdm(k_values, desc="Sweeping k values", disable=not verbose):
        _, top_k_indices = t.topk(all_feature_scores, k=k)

        result = train_sparse_probe_on_top_k(
            sae_activations_train,
            sae_activations_test,
            labels_train,
            labels_test,
            top_k_indices,
            seed,
            verbose=False,
        )

        mcc_values.append(result["mcc"])
        train_acc_values.append(result["train_acc"])
        test_acc_values.append(result["test_acc"])
        correlation_values.append(result["correlation"])

        if verbose:
            print(
                f"k={k:3d}: MCC={result['mcc']:.4f}, Test Acc={result['test_acc']:.4f}"
            )

    return {
        "k_values": np.array(k_values),
        "mcc_values": np.array(mcc_values),
        "train_acc_values": np.array(train_acc_values),
        "test_acc_values": np.array(test_acc_values),
        "correlation_values": np.array(correlation_values),
    }


def _sweep_k_multi_concept(
    sae_activations_train,
    sae_activations_test,
    labels_dict_train,
    labels_dict_test,
    all_feature_scores_dict,
    k_values,
    seed,
    verbose,
    compute_activation_mcc,
    mcc_union_features=False,
):
    """
    Helper for multi-concept k-sweep.

    Trains one probe per concept and computes:
    1. Probe-label correlation for each concept (for left & middle plots)
    2. Optionally: Activation-label MCC using compute_correlation_matrix (for right plot)

    Args:
        mcc_union_features: If True, use union of top-k features across concepts for MCC.
                        If False (default), compute MCC per concept and average.
    """
    results = {"k_values": np.array(k_values)}

    # Initialize storage for each concept
    for concept in labels_dict_train.keys():
        results[concept] = {
            "probe_correlation": [],
            "train_acc": [],
            "test_acc": [],
            "mcc": [],
        }

    aggregate_mcc_list = [] if compute_activation_mcc else None
    sae_acts_test_tensor = (
        t.tensor(sae_activations_test, dtype=t.float32)
        if compute_activation_mcc
        else None
    )

    for k in tqdm(
        k_values, desc="Sweeping k for all concepts", disable=not verbose
    ):
        concept_act_mccs = []
        all_top_k_indices_set = set() if mcc_union_features else None

        for concept in labels_dict_train.keys():
            # Get top-k features for this concept
            all_scores = all_feature_scores_dict[concept]
            if isinstance(all_scores, np.ndarray):
                all_scores = t.tensor(all_scores)
            _, top_k_indices = t.topk(all_scores, k=k)

            # Compute activation-label MCC
            if compute_activation_mcc:
                if mcc_union_features:
                    # Collect indices for union method
                    all_top_k_indices_set.update(top_k_indices.tolist())
                else:
                    # Per-concept method: compute MCC using only this concept's top-k features
                    sae_acts_test_concept = sae_acts_test_tensor[
                        :, top_k_indices
                    ]
                    single_concept_labels = {
                        concept: labels_dict_test[concept]
                    }

                    corr_matrix_concept, _ = compute_correlation_matrix(
                        sae_acts_test_concept, single_concept_labels
                    )

                    # Take max absolute correlation across this concept's top-k features
                    max_corr_this_concept = (
                        corr_matrix_concept.abs().max().item()
                    )
                    concept_act_mccs.append(max_corr_this_concept)

            # Train probe
            result = train_sparse_probe_on_top_k(
                sae_activations_train,
                sae_activations_test,
                labels_dict_train[concept],
                labels_dict_test[concept],
                top_k_indices,
                seed,
                verbose=False,
            )

            # Store probe-based metrics
            results[concept]["probe_correlation"].append(result["correlation"])
            results[concept]["train_acc"].append(result["train_acc"])
            results[concept]["test_acc"].append(result["test_acc"])
            results[concept]["mcc"].append(result["mcc"])

            if verbose:
                print(
                    f"k={k:3d}, {concept}: Probe Corr={result['correlation']:.4f}"
                )

        # Compute aggregate activation MCC
        if compute_activation_mcc:
            if mcc_union_features:
                # Union method: use all top-k features across concepts
                all_top_k_indices = sorted(list(all_top_k_indices_set))
                sae_acts_test_union = sae_acts_test_tensor[
                    :, all_top_k_indices
                ]

                # Compute correlation matrix: (features, concepts)
                corr_matrix_union, _ = compute_correlation_matrix(
                    sae_acts_test_union, labels_dict_test
                )

                # For each concept, take max correlation across all union features
                max_corr_per_concept = (
                    corr_matrix_union.abs().max(dim=0).values
                )
                aggregate_mcc = max_corr_per_concept.mean().item()
            else:
                # Per-concept method: average the per-concept MCCs
                aggregate_mcc = np.mean(concept_act_mccs)

            aggregate_mcc_list.append(aggregate_mcc)

    # Convert lists to arrays
    for concept in labels_dict_train.keys():
        for key in results[concept]:
            results[concept][key] = np.array(results[concept][key])

    if compute_activation_mcc:
        results["aggregate_activation_mcc"] = np.array(aggregate_mcc_list)

    return results


def get_probe_logits_with_intervention(
    model,
    tokenizer,
    submodule_steer_name,
    submodule_probe_name,
    dictionary,
    probe,
    sentences,
    intervention_indices=None,
    intervention_type="zero",
    intervention_strength=1.0,
    use_sparsemax=False,
    batch_size=32,
    scaler=None,
    multiclass=False,
):
    """
    Get probe logits with optional intervention on SAE features.

    Args:
        model: Language model
        tokenizer: Tokenizer
        submodule_steer_name: Name of submodule to intervene on
        submodule_probe_name: Name of submodule to extract probe activations
        dictionary: SAE model
        probe: Trained sklearn probe (binary or multiclass)
        sentences: List of sentences
        intervention_indices: Indices of SAE features to intervene on (None = no intervention)
                            For "add_decoder", uses only the first index (top feature)
        intervention_type: Type of intervention:
                        - "zero": Set features to 0
                        - "amplify": Multiply features by intervention_strength
                        - "ablate": Same as zero
                        - "add_decoder": Add decoder direction to residual stream
        intervention_strength: Strength of intervention (for amplify/add_decoder, default: 2.0)
        use_sparsemax: Whether SAE uses sparsemax
        batch_size: Batch size for processing
        scaler: Optional StandardScaler for normalizing activations
        multiclass: If True, return all class logits (n_samples, n_classes).
                If False, return binary logits (n_samples,) - squeezes if needed.

    Returns:
        logits: Probe logits
                - If multiclass=False: (n_samples,) for binary probes
                - If multiclass=True: (n_samples, n_classes) for multiclass probes
    """
    submodule_steer = get_submodule_with_index(model, submodule_steer_name)
    submodule_probe = get_submodule_with_index(model, submodule_probe_name)

    all_logits = []

    # Process in batches
    num_batches = (len(sentences) + batch_size - 1) // batch_size

    for idx in range(num_batches):
        start_idx = idx * batch_size
        end_idx = min(start_idx + batch_size, len(sentences))
        batch = sentences[start_idx:end_idx]

        cache = {}

        def _set_sae_activation_with_intervention(_module, _input, _output):
            """Hook to replace activations with SAE reconstruction + intervention."""
            if isinstance(_output, tuple):
                resid_stream = _output[0]
            else:
                resid_stream = _output

            # Process through SAE
            if use_sparsemax:
                x_hat_flat, f_flat = dictionary(
                    resid_stream.flatten(start_dim=0, end_dim=1)
                )
                x_hat = x_hat_flat.view(
                    resid_stream.shape[0], resid_stream.shape[1], -1
                )
                f = f_flat.view(
                    resid_stream.shape[0], resid_stream.shape[1], -1
                )
            else:
                x_hat, f = dictionary(resid_stream)

            # INTERVENTION: Modify SAE features or add steering vectors
            if intervention_indices is not None:
                if intervention_type == "zero":
                    # Zero out specific features
                    f[:, :, intervention_indices] = 0.0
                elif intervention_type == "amplify":
                    # Amplify specific features
                    f[:, :, intervention_indices] *= intervention_strength
                elif intervention_type == "ablate":
                    # Same as zero
                    f[:, :, intervention_indices] = 0.0
                elif intervention_type == "add_decoder":
                    # Add decoder direction: take top feature and add its decoder column
                    # intervention_indices should be a single feature index for this mode
                    if isinstance(intervention_indices, list):
                        top_feature_idx = intervention_indices[
                            0
                        ]  # Use first/top feature
                    else:
                        top_feature_idx = intervention_indices

                    # Get the decoder column (direction) for this feature
                    # dictionary.decoder.weight shape: (rep_dim, hid_dim)
                    # We want column top_feature_idx: (rep_dim,)
                    decoder_direction = dictionary.decoder.weight[
                        :, top_feature_idx
                    ]  # (rep_dim,)

                    # Don't modify f, will add steering to reconstruction directly
                    pass

            # Reconstruct with intervened features
            if use_sparsemax:
                x_recon_flat = (
                    dictionary.decoder(f.flatten(start_dim=0, end_dim=1))
                    + dictionary.decoder.bias
                )
                x_recon = x_recon_flat.view(
                    resid_stream.shape[0], resid_stream.shape[1], -1
                )
            else:
                x_recon = dictionary.decoder(f) + dictionary.decoder.bias

            # Add steering vector if using add_decoder intervention
            if (
                intervention_indices is not None
                and intervention_type == "add_decoder"
            ):
                if isinstance(intervention_indices, list):
                    top_feature_idx = intervention_indices[0]
                else:
                    top_feature_idx = intervention_indices
                decoder_direction = dictionary.decoder.weight[
                    :, top_feature_idx
                ]

                # Add steering: x_recon = x_hat + intervention_strength * decoder_direction
                # Broadcast across batch and sequence dimensions
                steering_vector = (
                    intervention_strength * decoder_direction
                )  # (rep_dim,)
                x_recon = x_recon + steering_vector.unsqueeze(0).unsqueeze(
                    0
                )  # (1, 1, rep_dim)

            if isinstance(_output, tuple):
                _output = (_output[0].clone(),) + _output[1:]
                _output[0][:] = x_recon
            else:
                _output = x_recon

            return _output

        def _get_activations(_module, _input, _output):
            """Hook to capture activations for the probe."""
            if isinstance(_output, tuple):
                assert len(_output) == 1
                _output = _output[0]
            cache["x"] = _output
            return None

        # Register hooks
        hook_steer = submodule_steer.register_forward_hook(
            _set_sae_activation_with_intervention
        )
        hook_probe = submodule_probe.register_forward_hook(_get_activations)

        # Forward pass (no gradients needed)
        with t.no_grad():
            batch_tokenized = tokenizer(
                batch, return_tensors="pt", padding=True
            ).to(model.device)
            _ = model(**batch_tokenized)

        # Remove hooks
        hook_steer.remove()
        hook_probe.remove()

        # Process probe activations
        submod_acts = cache["x"].sum(dim=1).squeeze(dim=-1)

        # Normalize if scaler is provided
        if scaler is not None:
            original_dtype = submod_acts.dtype
            submod_acts_np = (
                submod_acts.cpu().float().numpy()
            )  # Convert to float32 for sklearn
            submod_acts_np = scaler.transform(submod_acts_np)
            submod_acts = t.tensor(submod_acts_np, dtype=original_dtype).to(
                model.device
            )  # Restore original dtype

        # Get probe logits (sklearn needs float32/float64)
        logits = probe.decision_function(submod_acts.cpu().float().numpy())

        # For binary probes (2 classes), decision_function returns (n_samples,)
        # When multiclass=True, we need to expand to (n_samples, 2) with both class logits
        if multiclass and logits.ndim == 1:
            # Binary probe: expand to 2-class format
            # Class 0 logit: -logits, Class 1 logit: +logits
            logits = np.stack([-logits, logits], axis=1)

        all_logits.append(logits)

    result = np.concatenate(all_logits, axis=0)

    # Handle binary vs multiclass output
    if not multiclass and result.ndim > 1:
        # User wants binary output but probe is multiclass: squeeze
        result = result.squeeze()

    return result


def compute_causal_intervention_matrix(
    model,
    tokenizer,
    submodule_steer_name,
    submodule_probe_name,
    dictionary,
    probes_dict,
    top_features_dict,
    sentences,
    labels_dict=None,
    intervention_type="zero",
    intervention_strength=1.0,
    use_sparsemax=False,
    batch_size=32,
    scaler=None,
):
    """
    Compute causal intervention matrix showing effect of steering one concept on another.

    For each pair (steer_concept, eval_concept):
    1. Get baseline logits for eval_concept (no intervention)
    2. Intervene on steer_concept's top features
    3. Get new logits for eval_concept
    4. Compute ΔLogOdds = logits_after - logits_before

    Args:
        model: Language model
        tokenizer: Tokenizer
        submodule_steer_name: Submodule name for steering
        submodule_probe_name: Submodule name for probe
        dictionary: SAE model
        probes_dict: Dict mapping concept name -> trained probe
        top_features_dict: Dict mapping concept name -> top-k feature indices
        sentences: List of sentences to evaluate
        labels_dict: Optional dict of labels for filtering/analysis
        intervention_type: "zero", "amplify", "ablate", or "add_decoder" (default: "add_decoder")
        intervention_strength: Strength for amplify/add_decoder intervention (default: 2.0)
        use_sparsemax: Whether SAE uses sparsemax
        batch_size: Batch size
        scaler: Optional StandardScaler

    Returns:
        delta_logodds_matrix: (num_concepts, num_concepts) numpy array
            delta_logodds_matrix[i, j] = effect of steering concept i on eval concept j
    """
    concepts = list(probes_dict.keys())
    num_concepts = len(concepts)
    delta_logodds_matrix = np.zeros((num_concepts, num_concepts))

    print("\nComputing causal intervention matrix...")
    print(f"Intervention type: {intervention_type}")
    if intervention_type == "amplify":
        print(f"Intervention strength: {intervention_strength}")

    # First, get baseline logits for all concepts (no intervention)
    baseline_logits_dict = {}
    print("\nGetting baseline logits (no intervention)...")
    for eval_concept in tqdm(concepts, desc="Baseline"):
        baseline_logits = get_probe_logits_with_intervention(
            model=model,
            tokenizer=tokenizer,
            submodule_steer_name=submodule_steer_name,
            submodule_probe_name=submodule_probe_name,
            dictionary=dictionary,
            probe=probes_dict[eval_concept],
            sentences=sentences,
            intervention_indices=None,  # No intervention
            use_sparsemax=use_sparsemax,
            batch_size=batch_size,
            scaler=scaler,
        )
        baseline_logits_dict[eval_concept] = baseline_logits

    # Now compute interventions
    print("\nComputing interventions...")
    for steer_idx, steer_concept in enumerate(
        tqdm(concepts, desc="Steer concept")
    ):
        # Get intervention indices for this steering concept
        intervention_indices = top_features_dict[steer_concept]
        if isinstance(intervention_indices, t.Tensor):
            intervention_indices = intervention_indices.tolist()

        for eval_idx, eval_concept in enumerate(concepts):
            # Get logits with intervention
            intervention_logits = get_probe_logits_with_intervention(
                model=model,
                tokenizer=tokenizer,
                submodule_steer_name=submodule_steer_name,
                submodule_probe_name=submodule_probe_name,
                dictionary=dictionary,
                probe=probes_dict[eval_concept],
                sentences=sentences,
                intervention_indices=intervention_indices,
                intervention_type=intervention_type,
                intervention_strength=intervention_strength,
                use_sparsemax=use_sparsemax,
                batch_size=batch_size,
                scaler=scaler,
            )

            # Compute ΔLogOdds
            delta = intervention_logits - baseline_logits_dict[eval_concept]
            delta_logodds_matrix[steer_idx, eval_idx] = delta.mean()

    return delta_logodds_matrix


def compute_causal_intervention_matrix_multiclass(
    model,
    tokenizer,
    submodule_steer_name,
    submodule_probe_name,
    dictionary,
    probes_dict,
    multiclass_groups,
    top_features_dict,
    feature_signs_dict,
    sentences,
    intervention_type="add_decoder",
    intervention_strength=2.0,
    use_sparsemax=False,
    batch_size=32,
    scaler=None,
):
    """
    Compute causal intervention matrix for multiclass probes.

    For each pair (steer_class, eval_class):
    1. Get baseline logits for eval_class's group (no intervention)
    2. Intervene on steer_class's top feature (with correct sign)
    3. Get new logits for eval_class's group
    4. Extract specific class logit
    5. Compute ΔLogOdds = logits_after - logits_before

    Args:
        model: Language model
        tokenizer: Tokenizer
        submodule_steer_name: Submodule name for steering
        submodule_probe_name: Submodule name for probe
        dictionary: SAE model
        probes_dict: Dict mapping group_name -> multiclass probe
        multiclass_groups: Dict with group info (class_names, concept_names, labels)
        top_features_dict: Dict mapping (group_name, class_idx) -> feature_idx
        feature_signs_dict: Dict mapping (group_name, class_idx) -> sign (+1 or -1)
        sentences: List of sentences to evaluate
        intervention_type: "add_decoder" (only this makes sense with signs)
        intervention_strength: Base strength (will be multiplied by sign)
        use_sparsemax: Whether SAE uses sparsemax
        batch_size: Batch size
        scaler: Optional StandardScaler

    Returns:
        delta_logodds_matrix: (num_total_classes, num_total_classes) numpy array
        class_names_flat: List of full class names corresponding to matrix indices
    """
    # Flatten all classes across all groups
    class_names_flat = []
    class_to_group = {}  # class_flat_idx -> (group_name, class_idx)

    flat_idx = 0
    for group_name, group_info in multiclass_groups.items():
        for class_idx, concept_name in enumerate(group_info["concept_names"]):
            class_names_flat.append(concept_name)
            class_to_group[flat_idx] = (group_name, class_idx)
            flat_idx += 1

    num_total_classes = len(class_names_flat)
    delta_logodds_matrix = np.zeros((num_total_classes, num_total_classes))

    print(
        f"\nComputing multiclass intervention matrix ({num_total_classes} x {num_total_classes})..."
    )
    print(f"Intervention type: {intervention_type}")
    print(f"Base strength: {intervention_strength}")

    # Get baseline logits for all groups (no intervention)
    baseline_logits_dict = (
        {}
    )  # group_name -> logits array (n_samples, n_classes)
    print("\nGetting baseline logits (no intervention)...")
    for group_name, probe in tqdm(probes_dict.items(), desc="Baseline"):
        baseline_logits = get_probe_logits_with_intervention(
            model=model,
            tokenizer=tokenizer,
            submodule_steer_name=submodule_steer_name,
            submodule_probe_name=submodule_probe_name,
            dictionary=dictionary,
            probe=probe,
            sentences=sentences,
            intervention_indices=None,  # No intervention
            use_sparsemax=use_sparsemax,
            batch_size=batch_size,
            scaler=scaler,
            multiclass=True,  # Return all class logits
        )
        baseline_logits_dict[group_name] = baseline_logits

    # Compute interventions
    print("\nComputing interventions...")
    for steer_flat_idx in tqdm(range(num_total_classes), desc="Steer class"):
        steer_group, steer_class_idx = class_to_group[steer_flat_idx]
        steer_feature = top_features_dict[(steer_group, steer_class_idx)]
        steer_sign = feature_signs_dict[(steer_group, steer_class_idx)]

        # Adjust intervention strength by sign
        signed_strength = intervention_strength * steer_sign

        for eval_flat_idx in range(num_total_classes):
            eval_group, eval_class_idx = class_to_group[eval_flat_idx]

            # Get logits with intervention
            intervention_logits = get_probe_logits_with_intervention(
                model=model,
                tokenizer=tokenizer,
                submodule_steer_name=submodule_steer_name,
                submodule_probe_name=submodule_probe_name,
                dictionary=dictionary,
                probe=probes_dict[eval_group],
                sentences=sentences,
                intervention_indices=steer_feature,
                intervention_type=intervention_type,
                intervention_strength=signed_strength,  # Use signed strength
                use_sparsemax=use_sparsemax,
                batch_size=batch_size,
                scaler=scaler,
                multiclass=True,  # Return all class logits
            )

            # Extract specific class logits
            # intervention_logits shape: (n_samples, n_classes_in_group)
            # We want logits for eval_class_idx
            if intervention_logits.ndim == 1:
                # Binary case, already scalar
                eval_logits_after = intervention_logits
                eval_logits_before = baseline_logits_dict[eval_group]
            else:
                # Multiclass case, extract column
                eval_logits_after = intervention_logits[:, eval_class_idx]
                eval_logits_before = baseline_logits_dict[eval_group][
                    :, eval_class_idx
                ]

            # Compute ΔLogOdds
            delta = eval_logits_after - eval_logits_before
            delta_logodds_matrix[steer_flat_idx, eval_flat_idx] = delta.mean()

    return delta_logodds_matrix, class_names_flat


def group_concepts_into_multiclass(labels_dict):
    """
    Group binary concept labels into multiclass groups.

    Examples:
        tense-present, tense-past → tense: [0, 1]
        sentiment-positive, sentiment-neutral, sentiment-negative → sentiment: [0, 1, 2]

    Args:
        labels_dict: Dict of {concept_name: binary_labels}

    Returns:
        multiclass_groups: Dict of {group_name: {
            'class_names': [class1, class2, ...],
            'labels': multiclass_labels  # 0, 1, 2, ... for each class
        }}
    """
    from collections import defaultdict

    # Group concepts by prefix
    groups = defaultdict(list)
    for concept_name in labels_dict.keys():
        if "-" in concept_name:
            prefix, suffix = concept_name.rsplit("-", 1)
            groups[prefix].append((suffix, concept_name))
        else:
            # Standalone concept - treat as its own binary group
            groups[concept_name].append(("positive", concept_name))

    # Convert to multiclass format
    multiclass_groups = {}
    for group_name, class_list in groups.items():
        class_names = [suffix for suffix, _ in class_list]
        concept_names = [full_name for _, full_name in class_list]

        # Create multiclass labels
        num_samples = len(labels_dict[concept_names[0]])
        multiclass_labels = np.zeros(num_samples, dtype=int)

        for class_idx, concept_name in enumerate(concept_names):
            # Where this binary label is True, set multiclass label to class_idx
            binary_labels = np.array(labels_dict[concept_name], dtype=bool)
            multiclass_labels[binary_labels] = class_idx

        multiclass_groups[group_name] = {
            "class_names": class_names,
            "concept_names": concept_names,
            "labels": multiclass_labels,
        }

    return multiclass_groups


def run_causal_intervention_experiment(args):
    """
    Run standalone causal intervention experiment with multiclass probes.

    This computes the intervention matrix showing how steering one concept class
    affects another concept class's logits.
    """
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from pathlib import Path

    print("=" * 70)
    print("CAUSAL INTERVENTION EXPERIMENT (Multiclass Probes)")
    print("=" * 70)

    # Load paired sentences and labels for difference-based SAE
    # Use unified pairs across all concepts for this experiment
    sentences_1, sentences_2, labels = load_paired_sentences_test_all_concepts(seed=args.seed)
    print(
        f"\nLoaded {len(sentences_1)} sentence pairs with {len(labels)} binary concepts"
    )

    # Limit number of samples if specified
    if args.max_samples is not None and args.max_samples < len(sentences_1):
        print(
            f"Limiting to {args.max_samples} sample pairs (from {len(sentences_1)})"
        )
        indices = np.random.RandomState(args.seed).choice(
            len(sentences_1), args.max_samples, replace=False
        )
        sentences_1 = [sentences_1[i] for i in indices]
        sentences_2 = [sentences_2[i] for i in indices]
        labels = {k: [v[i] for i in indices] for k, v in labels.items()}
        print(f"Using {len(sentences_1)} sentence pairs")

    # Filter concepts if specified
    if args.concepts:
        labels = {k: v for k, v in labels.items() if k in args.concepts}
        print(f"Filtered to {len(labels)} concepts: {list(labels.keys())}")

    # Group into multiclass problems
    print("\nGrouping concepts into multiclass problems...")
    multiclass_groups = group_concepts_into_multiclass(labels)
    for group_name, group_info in multiclass_groups.items():
        print(
            f"  {group_name}: {len(group_info['class_names'])} classes - {group_info['class_names']}"
        )
    print(f"Total multiclass groups: {len(multiclass_groups)}")

    # Load or compute SAE activations (for probe training on residual stream)
    if args.sae_activations_path:
        print(
            f"\nLoading pre-computed SAE activations from {args.sae_activations_path}"
        )
        if str(args.sae_activations_path).endswith(".npy"):
            sae_activations = np.load(args.sae_activations_path)
        else:
            sae_activations = t.load(args.sae_activations_path).numpy()
        print(f"SAE activations shape: {sae_activations.shape}")

    # Split train/test using the labels of the first multiclass group for stratification
    print("\nSplitting train/test...")
    first_group = list(multiclass_groups.keys())[0]
    train_idx, test_idx = train_test_split(
        np.arange(len(sentences_1)),
        test_size=0.2,
        random_state=args.seed,
        stratify=multiclass_groups[first_group]["labels"],
    )

    sentences_1_train = [sentences_1[i] for i in train_idx]
    sentences_2_train = [sentences_2[i] for i in train_idx]
    sentences_1_test = [sentences_1[i] for i in test_idx]
    sentences_2_test = [sentences_2[i] for i in test_idx]

    # Split multiclass labels into train/test
    multiclass_train = {}
    multiclass_test = {}
    for group_name, group_info in multiclass_groups.items():
        multiclass_train[group_name] = {
            "class_names": group_info["class_names"],
            "concept_names": group_info["concept_names"],
            "labels": group_info["labels"][train_idx],
        }
        multiclass_test[group_name] = {
            "class_names": group_info["class_names"],
            "concept_names": group_info["concept_names"],
            "labels": group_info["labels"][test_idx],
        }

    print(f"Train: {len(train_idx)}, Test: {len(test_idx)}")

    # Load language model
    print(f"\nLoading language model: {args.lm_model_name}")
    lm_model = AutoModelForCausalLM.from_pretrained(
        args.lm_model_name,
        torch_dtype=t.float32,
        device_map="cuda",
        low_cpu_mem_usage=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.lm_model_name
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # Load SAE dictionary
    sae_model = load_model(args.model_path).to("cuda")
    # Keep SAE in float32 for better numerical precision

    # Determine submodule names
    if args.submodule_steer is None:
        if "pythia" in args.lm_model_name:
            args.submodule_steer = f"gpt_neox.layers.{sae_model.layer}"
        elif "gemma" in args.lm_model_name:
            args.submodule_steer = f"model.layers.{sae_model.layer}"
        else:
            raise ValueError(
                "--submodule-steer must be specified for this model"
            )

    if args.submodule_probe is None:
        args.submodule_probe = args.submodule_steer

    print(f"Submodule steer: {args.submodule_steer}")
    print(f"Submodule probe: {args.submodule_probe}")

    # Determine which concepts/groups are needed early for optimization
    concepts_needed = set()
    if args.intervention_output:
        # Heatmap needs all concepts
        for group_name, group_info in multiclass_train.items():
            for concept_name in group_info["concept_names"]:
                concepts_needed.add(concept_name)
    elif args.steering_curves:
        # Only need concepts in steering curves
        for pair_str in args.steering_curves.split(";"):
            concepts = pair_str.strip().split(",")
            if len(concepts) == 2:
                concepts_needed.add(concepts[0].strip())
                concepts_needed.add(concepts[1].strip())
    else:
        # Neither specified, need all
        for group_name, group_info in multiclass_train.items():
            for concept_name in group_info["concept_names"]:
                concepts_needed.add(concept_name)

    # Determine which groups we need to train probes for
    groups_needed = set()
    for group_name, group_info in multiclass_train.items():
        for concept_name in group_info["concept_names"]:
            if concept_name in concepts_needed:
                groups_needed.add(group_name)
                break  # Only need to find one match per group

    print(
        f"\nOptimization: Training probes for {len(groups_needed)}/{len(multiclass_train)} groups"
    )
    print(f"Groups needed: {sorted(groups_needed)}")
    print(f"Concepts needed: {sorted(concepts_needed)}")

    # Helper: extract residual stream activations for a list of sentences
    def extract_residual_activations(
        model, tokenizer, submodule, sentences, batch_size
    ):
        residual_acts_list = []

        def _capture_residual(_module, _input, _output):
            if isinstance(_output, tuple):
                _output = _output[0]
            # Sum over sequence dimension to get per-example vector
            residual_acts_list.append(_output.sum(dim=1).detach().cpu())

        hook = submodule.register_forward_hook(_capture_residual)
        with t.no_grad():
            for i in tqdm(
                range(0, len(sentences), batch_size),
                desc="Extracting residual activations",
            ):
                batch = sentences[i : i + batch_size]
                batch_tokenized = tokenizer(
                    batch, return_tensors="pt", padding=True
                ).to(model.device)
                _ = model(**batch_tokenized)
        hook.remove()

        return t.cat(residual_acts_list, dim=0).cpu().numpy()

    # Extract residual stream activations for train and test
    print(
        "\nExtracting residual stream activations from language model (TRAIN)..."
    )
    submodule_probe = get_submodule_with_index(lm_model, args.submodule_probe)
    residual_acts_train = extract_residual_activations(
        lm_model,
        tokenizer,
        submodule_probe,
        sentences_train,
        args.batch_size,
    )
    print(
        f"Residual stream TRAIN activations shape: {residual_acts_train.shape}"
    )

    print(
        "\nExtracting residual stream activations from language model (TEST)..."
    )
    residual_acts_test = extract_residual_activations(
        lm_model,
        tokenizer,
        submodule_probe,
        sentences_test,
        args.batch_size,
    )
    print(
        f"Residual stream TEST activations shape: {residual_acts_test.shape}"
    )

    # Normalize activations (fit on train, apply to both train and test)
    scaler = StandardScaler()
    residual_acts_train_normalized = scaler.fit_transform(residual_acts_train)
    residual_acts_test_normalized = scaler.transform(residual_acts_test)
    print(
        f"Normalized residual TRAIN activations (mean={residual_acts_train_normalized.mean():.4f}, std={residual_acts_train_normalized.std():.4f})"
    )

    # Train multiclass probes for each group (only for needed groups), using train_probes
    print("\nTraining multiclass probes on residual stream...")
    probes_dict = {}  # group_name -> probe
    for group_name, group_info in multiclass_train.items():
        # Skip groups we don't need
        if group_name not in groups_needed:
            print(f"  {group_name}: Skipped (not needed)")
            continue

        print(
            f"  Training probe for group '{group_name}' ({len(group_info['class_names'])} classes)..."
        )

        # Use shared train_probes helper in multiclass mode
        probe = train_probes(
            train_activations=residual_acts_train_normalized,
            train_labels=group_info["labels"],
            test_activations=residual_acts_test_normalized,
            test_labels=multiclass_test[group_name]["labels"],
            probe_type="multiclass",
            seed=args.seed,
            sparse=None,  # use all features; no top-k preselection here
        )

        probes_dict[group_name] = probe

    # Get top-1 feature for each class using gradient attribution
    print(
        f"\nComputing gradient attribution for {len(concepts_needed)} concepts..."
    )
    print(f"Concepts: {sorted(concepts_needed)}")
    top_features_dict = {}  # (group_name, class_idx) -> feature_idx
    feature_signs_dict = {}  # (group_name, class_idx) -> sign (+1 or -1)

    for group_name, group_info in multiclass_train.items():
        # Skip if we didn't train a probe for this group
        if group_name not in probes_dict:
            continue

        probe = probes_dict[group_name]
        for class_idx, class_name in enumerate(group_info["class_names"]):
            full_name = group_info["concept_names"][class_idx]

            # Skip if this concept is not needed
            if full_name not in concepts_needed:
                continue

            print(f"  {full_name} (class {class_idx} of {group_name})...")

            _, top_indices, all_scores = find_top_k_features_by_attribution_pairs(
                model=lm_model,
                tokenizer=tokenizer,
                submodule_steer_name=args.submodule_steer,
                submodule_probe_name=args.submodule_probe,
                dictionary=sae_model,
                probe=probe,
                sentences_1=sentences_1_train,
                sentences_2=sentences_2_train,
                k=1,  # Only need top-1 for intervention
                use_sparsemax=args.use_sparsemax,
                batch_size=args.batch_size,
                scaler=scaler,
                logit_idx=class_idx,  # Which class logit to compute gradients for
            )
            top_feature_idx = top_indices.item()
            top_features_dict[(group_name, class_idx)] = top_feature_idx

            # Determine sign: does adding decoder direction increase this class's logit?
            # Method: compute dot product of decoder direction with probe weights for this class
            decoder_vec = (
                sae_model.decoder.weight[:, top_feature_idx]
                .detach()
                .cpu()
                .numpy()
            )

            if probe.coef_.shape[0] == 1:
                # Binary probe: class 0 uses -coef_[0], class 1 uses +coef_[0]
                probe_weights_for_class = (
                    probe.coef_[0] if class_idx == 1 else -probe.coef_[0]
                )
            else:
                # Multiclass probe: class i uses coef_[i]
                probe_weights_for_class = probe.coef_[class_idx]

            # Sign: positive if decoder direction aligns with probe weights
            dot_product = np.dot(decoder_vec, probe_weights_for_class)
            feature_signs_dict[(group_name, class_idx)] = (
                1.0 if dot_product > 0 else -1.0
            )

            print(
                f"    Top feature: {top_feature_idx} (sign: {feature_signs_dict[(group_name, class_idx)]:+.0f}, dot={dot_product:.2f})"
            )

    # Compute causal intervention matrix (only if needed for heatmap or if no steering curves requested)
    if args.intervention_output or not args.steering_curves:
        delta_logodds_matrix, class_names_flat = (
            compute_causal_intervention_matrix_multiclass(
                model=lm_model,
                tokenizer=tokenizer,
                submodule_steer_name=args.submodule_steer,
                submodule_probe_name=args.submodule_probe,
                dictionary=sae_model,
                probes_dict=probes_dict,
                multiclass_groups=multiclass_test,
                top_features_dict=top_features_dict,
                feature_signs_dict=feature_signs_dict,
                sentences=sentences_test,
                intervention_type=args.intervention_type,
                intervention_strength=args.intervention_strength,
                use_sparsemax=args.use_sparsemax,
                batch_size=args.batch_size,
                scaler=scaler,
            )
        )
    else:
        print(
            "\nSkipping heatmap matrix computation (only generating steering curves)"
        )
        delta_logodds_matrix = None
        class_names_flat = None

    # Print results (only if matrix was computed)
    if delta_logodds_matrix is not None:
        print("\n" + "=" * 70)
        print("INTERVENTION MATRIX (ΔLogOdds)")
        print("=" * 70)

        # Print header
        print(f"{'Steer Eval':20s}", end="")
        for concept in class_names_flat:
            print(f"{concept:>20s}", end="")
        print()
        print("-" * (20 + 20 * len(class_names_flat)))

        # Print matrix rows
        for i, steer_concept in enumerate(class_names_flat):
            print(f"{steer_concept:20s}", end="")
            for j, eval_concept in enumerate(class_names_flat):
                print(f"{delta_logodds_matrix[i, j]:20.4f}", end="")
            print()

    # Save results if output path provided
    if args.intervention_output:
        matrix_json_path = args.intervention_output.with_suffix(".json")

        # Convert tuple keys to strings for JSON serialization
        top_features_serializable = {
            f"{k[0]}_{k[1]}": v for k, v in top_features_dict.items()
        }
        feature_signs_serializable = {
            f"{k[0]}_{k[1]}": v for k, v in feature_signs_dict.items()
        }

        save_data = {
            "intervention_type": args.intervention_type,
            "intervention_strength": args.intervention_strength,
            "class_names": class_names_flat,
            "multiclass_groups": {
                group_name: {
                    "class_names": info["class_names"],
                    "concept_names": info["concept_names"],
                }
                for group_name, info in multiclass_test.items()
            },
            "top_features": top_features_serializable,
            "feature_signs": feature_signs_serializable,
            "delta_logodds_matrix": delta_logodds_matrix.tolist(),
        }
        with open(matrix_json_path, "w") as f:
            json.dump(save_data, f, indent=2)
        print(f"\nIntervention matrix data saved to {matrix_json_path}")

        plot_causal_intervention_matrix(
            delta_logodds_matrix,
            class_names_flat,
            output_path=args.intervention_output,
        )
        print(f"Intervention matrix plot saved to {args.intervention_output}")

    # Optionally generate steering strength curves for specified concept pairs
    if args.steering_curves:
        print("\n" + "=" * 70)
        print("GENERATING STEERING STRENGTH CURVES")
        print("=" * 70)

        # Default strength values to test
        strength_values = [0.1, 0.5, 1.0, 2.0, 5.0]

        # Parse concept pairs from argument (format: "concept1,concept2;concept3,concept4")
        concept_pairs = []
        for pair_str in args.steering_curves.split(";"):
            concepts = pair_str.strip().split(",")
            if len(concepts) == 2:
                concept_pairs.append(
                    (concepts[0].strip(), concepts[1].strip())
                )

        # Extract model name from model path (e.g., "labeled-sentences_seed0" from path)
        model_name = args.model_path.name

        # Determine base output path for curves
        if args.intervention_output:
            base_output = args.intervention_output
        else:
            base_output = Path(f"outputs/steering_curve_{model_name}.png")
            base_output.parent.mkdir(parents=True, exist_ok=True)

        for idx, (concept_i, concept_j) in enumerate(concept_pairs):
            if args.intervention_output:
                curve_output = base_output.with_name(
                    f"{base_output.stem}_{model_name}_curve_{idx+1}.png"
                )
            else:
                curve_output = base_output.with_name(
                    f"{base_output.stem}_curve_{idx+1}.png"
                )

            plot_steering_strength_curves(
                model=lm_model,
                tokenizer=tokenizer,
                submodule_steer_name=args.submodule_steer,
                submodule_probe_name=args.submodule_probe,
                dictionary=sae_model,
                probes_dict=probes_dict,
                multiclass_groups=multiclass_test,
                top_features_dict=top_features_dict,
                feature_signs_dict=feature_signs_dict,
                sentences=sentences_test,
                concept_pair=(concept_i, concept_j),
                strength_values=strength_values,
                intervention_type=args.intervention_type,
                use_sparsemax=args.use_sparsemax,
                batch_size=args.batch_size,
                scaler=scaler,
                output_path=curve_output,
            )


def plot_causal_intervention_matrix(
    delta_logodds_matrix,
    concept_names,
    output_path=None,
    title="Causal Intervention Matrix",
):
    """
    Plot the causal intervention heatmap.

    Args:
        delta_logodds_matrix: (n_concepts, n_concepts) matrix of ΔLogOdds
        concept_names: List of concept names
        output_path: Path to save plot (optional)
        title: Plot title
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    fig, ax = plt.subplots(figsize=(10, 8))

    # Robust normalization: clip to ±2.5σ to show lighter colors for weaker effects
    mean_val = delta_logodds_matrix.mean()
    std_val = delta_logodds_matrix.std()
    vmax = mean_val + 2.5 * std_val
    vmin = mean_val - 2.5 * std_val

    # Ensure symmetric around zero for diverging colormap
    vabs = max(abs(vmin), abs(vmax))
    vmin, vmax = -vabs, vabs

    # Create heatmap (blue=positive, red=negative, no annotations)
    sns.heatmap(
        delta_logodds_matrix,
        annot=False,  # No numbers on heatmap
        cmap="RdBu",  # Red for negative, Blue for positive (not reversed)
        center=0.0,
        vmin=vmin,  # Clip outliers
        vmax=vmax,
        xticklabels=concept_names,
        yticklabels=concept_names,
        cbar_kws={"label": "ΔLogOdds"},
        ax=ax,
        linewidths=0.5,  # Add gridlines
        linecolor="black",
    )

    # ax.set_xlabel("Eval Concept", fontsize=12)
    # ax.set_ylabel("Steer Concept", fontsize=12)
    # ax.set_title(title, fontsize=14)

    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved plot to {output_path}")

    return fig


def plot_steering_strength_curves(
    model,
    tokenizer,
    submodule_steer_name,
    submodule_probe_name,
    dictionary,
    probes_dict,
    multiclass_groups,
    top_features_dict,
    feature_signs_dict,
    sentences,
    concept_pair,
    strength_values,
    intervention_type="add_decoder",
    use_sparsemax=False,
    batch_size=32,
    scaler=None,
    output_path=None,
):
    """
    Plot ΔLogOdds vs steering strength for a pair of concepts.

    Recreates Figure 11 from the reference paper, showing:
    - Blue line: Steer z_i, measure z_i (should increase)
    - Red line: Steer z_j, measure z_i (should be flat/small if independent)
    - Purple solid: Steer both z_i and z_j, measure z_i
    - Purple dotted: Sum of individual effects (z_i + z_j)

    Args:
        concept_pair: Tuple of (concept_i_name, concept_j_name) e.g. ("tense-present", "sentiment-positive")
        strength_values: List of steering strengths to test (e.g., [0.1, 0.5, 1.0, 2.0, 5.0])
    """
    import matplotlib.pyplot as plt

    concept_i, concept_j = concept_pair

    # Find group and class indices for both concepts
    group_i, class_i = None, None
    group_j, class_j = None, None

    for group_name, group_info in multiclass_groups.items():
        for class_idx, concept_name in enumerate(group_info["concept_names"]):
            if concept_name == concept_i:
                group_i, class_i = group_name, class_idx
            if concept_name == concept_j:
                group_j, class_j = group_name, class_idx

    if group_i is None or group_j is None:
        raise ValueError(
            f"Concepts {concept_i} or {concept_j} not found in multiclass_groups"
        )

    # Get features and signs
    feature_i = top_features_dict[(group_i, class_i)]
    sign_i = feature_signs_dict[(group_i, class_i)]
    feature_j = top_features_dict[(group_j, class_j)]
    sign_j = feature_signs_dict[(group_j, class_j)]

    # Get baseline logits for concept_i
    baseline_logits = get_probe_logits_with_intervention(
        model=model,
        tokenizer=tokenizer,
        submodule_steer_name=submodule_steer_name,
        submodule_probe_name=submodule_probe_name,
        dictionary=dictionary,
        probe=probes_dict[group_i],
        sentences=sentences,
        intervention_indices=None,
        use_sparsemax=use_sparsemax,
        batch_size=batch_size,
        scaler=scaler,
        multiclass=True,
    )

    # Extract baseline for class_i
    if baseline_logits.ndim == 1:
        baseline_logits_i = baseline_logits
    else:
        baseline_logits_i = baseline_logits[:, class_i]

    baseline_mean = baseline_logits_i.mean()

    # Storage for results
    results = {
        "steer_i_measure_i": [],  # Blue line
        "steer_j_measure_i": [],  # Red line
        "steer_both_measure_i": [],  # Purple solid
        "steer_sum": [],  # Purple dotted
    }

    print(f"\nComputing steering curves for {concept_i} vs {concept_j}...")

    for strength in tqdm(strength_values, desc="Strength values"):
        # 1. Steer z_i, measure z_i (blue)
        logits_i = get_probe_logits_with_intervention(
            model=model,
            tokenizer=tokenizer,
            submodule_steer_name=submodule_steer_name,
            submodule_probe_name=submodule_probe_name,
            dictionary=dictionary,
            probe=probes_dict[group_i],
            sentences=sentences,
            intervention_indices=feature_i,
            intervention_type=intervention_type,
            intervention_strength=strength * sign_i,
            use_sparsemax=use_sparsemax,
            batch_size=batch_size,
            scaler=scaler,
            multiclass=True,
        )
        logits_i = logits_i[:, class_i] if logits_i.ndim > 1 else logits_i
        delta_i = logits_i.mean() - baseline_mean
        results["steer_i_measure_i"].append(delta_i)

        # 2. Steer z_j, measure z_i (red)
        logits_j = get_probe_logits_with_intervention(
            model=model,
            tokenizer=tokenizer,
            submodule_steer_name=submodule_steer_name,
            submodule_probe_name=submodule_probe_name,
            dictionary=dictionary,
            probe=probes_dict[group_i],
            sentences=sentences,
            intervention_indices=feature_j,
            intervention_type=intervention_type,
            intervention_strength=strength * sign_j,
            use_sparsemax=use_sparsemax,
            batch_size=batch_size,
            scaler=scaler,
            multiclass=True,
        )
        logits_j = logits_j[:, class_i] if logits_j.ndim > 1 else logits_j
        delta_j = logits_j.mean() - baseline_mean
        results["steer_j_measure_i"].append(delta_j)

        # 3. Sum for dotted purple line
        results["steer_sum"].append(delta_i + delta_j)

        # 4. Steer both (purple solid) - need to implement multi-feature steering
        # For now, approximate as sequential: steer i, then steer j on top
        # This is a simplification - ideally we'd steer both simultaneously
        results["steer_both_measure_i"].append(
            delta_i + delta_j
        )  # Placeholder

    # Create plot
    fig, ax = plt.subplots(figsize=(6, 4))

    ax.plot(
        strength_values,
        results["steer_i_measure_i"],
        marker="o",
        color="blue",
        label=f"{concept_i}",
        linewidth=2,
    )
    ax.plot(
        strength_values,
        results["steer_j_measure_i"],
        marker="o",
        color="red",
        label=f"{concept_j}",
        linewidth=2,
    )
    ax.plot(
        strength_values,
        results["steer_both_measure_i"],
        marker="o",
        color="purple",
        label=f"{concept_i} and {concept_j}",
        linewidth=2,
    )
    ax.plot(
        strength_values,
        results["steer_sum"],
        linestyle="--",
        marker=".",
        color="purple",
        alpha=0.7,
        label=f"{concept_i} + {concept_j}",
        linewidth=2,
    )

    ax.axhline(y=0, color="gray", linestyle="-", alpha=0.3, linewidth=1)
    ax.set_xlabel("Steering coefficient", fontsize=12)
    ax.set_ylabel(f"ΔLogOdds({concept_i})", fontsize=12)
    ax.legend(
        fontsize=10, loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2
    )
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved steering curve to {output_path}")

    return fig, results


def evaluate_sentence_labels(
    model_path: Path,
    threshold: float = 0.1,
    metrics: list = ["accuracy", "macrof1", "mcc"],
    batch_size: int = 128,
    k: int = None,
) -> Dict[str, Any]:
    """Evaluate SSAE on individual sentence labels with optional k-sparse probing."""

    # Load test sentences and labels
    sentences, labels = load_labeled_sentences_test()
    print(f"Loaded {len(sentences)} test sentences")
    print(f"Available labels: {list(labels.keys())}")

    # Load model first to get config
    model = load_model(model_path)
    print(f"Loaded model from {model_path}")
    print(
        f"Model: {model.model_name}, Layer: {model.layer}, Rep dim: {model.rep_dim}"
    )

    # Get sentence embeddings using model's config
    print("Extracting sentence embeddings...")
    embeddings = get_sentence_embeddings(
        sentences, model.model_name, model.layer, batch_size
    )
    print(f"Embeddings shape: {embeddings.shape}")

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
            # Get correlation matrix and top features
            corr_matrix, top_features = score_identification(
                activations, tensor_labels, threshold, metric
            )

            # If k is specified, train probes on top-k features and compute MCC
            if k is not None:
                print(
                    f"\nTraining probes on top-{k} features selected by correlation..."
                )
                from sklearn.model_selection import train_test_split

                probe_mcc_scores = {}
                activations_np = activations.numpy()

                # Use top_features dict keys to iterate (these match corr_matrix columns)
                for i, concept in enumerate(top_features.keys()):
                    # Get top-k features for this concept based on correlation
                    top_k_features = (
                        corr_matrix[:, i].abs().topk(k).indices.numpy()
                    )

                    # Split data
                    label_array = np.array(labels[concept])
                    if len(np.unique(label_array)) < 2:
                        continue

                    X_train, X_test, y_train, y_test = train_test_split(
                        activations_np[:, top_k_features],
                        label_array,
                        test_size=0.2,
                        random_state=42,
                        stratify=label_array,
                    )

                    # Train probe on top-k features
                    probe = LogisticRegression(
                        random_state=42, max_iter=1000, class_weight="balanced"
                    )
                    probe.fit(X_train, y_train)

                    # Compute MCC from probe predictions
                    logits = probe.decision_function(X_test)
                    probe_mcc = abs(pearsonr(logits, y_test)[0])
                    probe_mcc_scores[concept] = probe_mcc

                results[metric] = {
                    "correlation_matrix": corr_matrix,
                    "top_features": top_features,
                    "probe_mcc_k": probe_mcc_scores if k is not None else None,
                    "k": k,
                }
            else:
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


def infer_submodule_name(model_name, layer):
    """
    Infer the submodule name from model name and layer number.

    Args:
        model_name: HuggingFace model name (e.g., "EleutherAI/pythia-70m-deduped")
        layer: Layer number (int)

    Returns:
        Submodule name string (e.g., "gpt_neox.layers.5" or "model.layers.25")
    """
    if "pythia" in model_name.lower():
        return f"gpt_neox.layers.{layer}"
    elif "gemma" in model_name.lower():
        return f"model.layers.{layer}"
    else:
        raise ValueError(
            f"Unknown model architecture for {model_name}. "
            f"Please specify --submodule-steer and --submodule-probe manually."
        )


def get_submodule_with_index(model, submodule_name):
    """
    Get submodule, handling both attribute and index notation.

    This handles ModuleList indexing (e.g., model.layers[25] for Gemma)
    by parsing strings like "model.layers.25" and using index access when appropriate.

    Args:
        model: PyTorch model
        submodule_name: Dot-separated path (e.g., "model.layers.25")

    Returns:
        The requested submodule
    """
    parts = submodule_name.split(".")
    current = model
    for part in parts:
        if part.isdigit():
            # It's an index into a ModuleList/Sequential
            current = current[int(part)]
        else:
            # It's an attribute
            current = getattr(current, part)
    return current


def run_k_sweep_attribution(args):
    """
    Run k-sweep using LR coefficient-based feature selection.

    Uses trained logistic regression probe coefficients to rank features by importance.
    This method is much faster than gradient attribution and follows the
    heuristic_feature_ranking_binary approach with method="lr".

    This creates the three plots: domain=science correlation, sentiment=positive correlation, and aggregate MCC.
    """
    from sklearn.model_selection import train_test_split
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import matplotlib.pyplot as plt

    print("=" * 70)
    print("K-SWEEP WITH LR COEFFICIENT-BASED FEATURE SELECTION (PAIRWISE)")
    print("=" * 70)

    # Load paired sentences and labels for difference-based SAE
    sentences_1, sentences_2, labels = load_paired_sentences_test_all_concepts(seed=args.seed)
    print(f"\nLoaded {len(sentences_1)} sentence pairs")

    # Limit number of samples if specified (useful for faster debugging/testing with large models)
    if args.max_samples is not None and args.max_samples < len(sentences_1):
        print(
            f"Limiting to {args.max_samples} pairs (from {len(sentences_1)})"
        )
        indices = np.random.RandomState(args.seed).choice(
            len(sentences_1), args.max_samples, replace=False
        )
        sentences_1 = [sentences_1[i] for i in indices]
        sentences_2 = [sentences_2[i] for i in indices]
        labels = {k: [v[i] for i in indices] for k, v in labels.items()}
        print(f"Using {len(sentences_1)} sentence pairs")

    # Filter concepts if specified
    if args.concepts:
        labels = {k: v for k, v in labels.items() if k in args.concepts}
        print(f"Using concepts: {list(labels.keys())}")
    else:
        print(f"Using all {len(labels)} concepts")

    # Load or compute SAE activations (for pairs, compute delta activations)
    if args.sae_activations_path:
        print(
            f"\nLoading pre-computed SAE activations from {args.sae_activations_path}"
        )
        print("WARNING: Pre-computed activations should be SAE(delta_embeddings) = SAE(embeddings_2 - embeddings_1) for difference-based SAEs!")
        if str(args.sae_activations_path).endswith(".npy"):
            sae_activations = np.load(args.sae_activations_path)
        else:
            sae_activations = t.load(args.sae_activations_path).numpy()
        print(f"SAE delta activations shape: {sae_activations.shape}")
    else:
        print("\nComputing SAE activations from pairs...")
        sae_model = load_model(args.model_path)

        # Get embeddings for both sentences
        print("Extracting embeddings for first sentences...")
        embeddings_1 = get_sentence_embeddings(
            sentences_1, sae_model.model_name, sae_model.layer, args.batch_size
        )
        print("Extracting embeddings for second sentences...")
        embeddings_2 = get_sentence_embeddings(
            sentences_2, sae_model.model_name, sae_model.layer, args.batch_size
        )

        # For difference-based SAEs: compute delta embeddings FIRST, then pass through SAE
        # SAE(x2 - x1) != SAE(x2) - SAE(x1) because SAE is non-linear!
        print("Computing delta embeddings (embeddings_2 - embeddings_1)...")
        delta_embeddings = embeddings_2 - embeddings_1

        # Pass delta through the difference-based SAE
        print("Passing delta embeddings through SAE...")
        sae_activations = get_activations(sae_model, delta_embeddings).numpy()
        print(f"SAE activations shape: {sae_activations.shape}")

    # Split train/test
    print("\nSplitting train/test...")
    # Use first concept for stratification
    first_concept = list(labels.keys())[0]
    train_idx, test_idx = train_test_split(
        np.arange(len(sentences_1)),
        test_size=0.2,
        random_state=args.seed,
        stratify=labels[first_concept],
    )

    sae_train = sae_activations[train_idx]
    sae_test = sae_activations[test_idx]
    sentences_1_train = [sentences_1[i] for i in train_idx]
    sentences_2_train = [sentences_2[i] for i in train_idx]

    labels_train_dict = {
        concept: np.array(label_list)[train_idx]
        for concept, label_list in labels.items()
    }
    labels_test_dict = {
        concept: np.array(label_list)[test_idx]
        for concept, label_list in labels.items()
    }

    print(f"Train: {len(train_idx)}, Test: {len(test_idx)}")

    # Load SAE dictionary first to get model config
    sae_model = load_model(args.model_path).to("cuda")
    # Keep SAE in float32 for better numerical precision

    # Infer LM model name from SAE config if not provided
    if args.lm_model_name is None:
        args.lm_model_name = sae_model.model_name
        print(f"Inferred LM model from SAE config: {args.lm_model_name}")

    # Load language model
    print(f"\nLoading language model: {args.lm_model_name}")
    # Load in float32 for better numerical precision
    lm_model = AutoModelForCausalLM.from_pretrained(
        args.lm_model_name,
        torch_dtype=t.float32,  # Use float32 for better numerical precision
        device_map="cuda",  # Directly load to GPU instead of CPU->GPU transfer
        low_cpu_mem_usage=True,  # Reduce CPU memory usage during loading
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.lm_model_name
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # Determine submodule names - infer from SAE config if not provided
    if args.submodule_steer is None:
        args.submodule_steer = infer_submodule_name(
            sae_model.model_name, sae_model.layer
        )
        print(
            f"Inferred submodule from SAE config: {args.submodule_steer} "
            f"(model: {sae_model.model_name}, layer: {sae_model.layer})"
        )

    if args.submodule_probe is None:
        args.submodule_probe = args.submodule_steer

    print(f"Submodule steer: {args.submodule_steer}")
    print(f"Submodule probe: {args.submodule_probe}")

    # Extract residual stream activations from language model for probe training
    # For difference-based SAEs, extract activations from BOTH sentences and compute delta
    print(
        "\nExtracting residual stream activations from language model (paired sentences)..."
    )
    submodule_probe = get_submodule_with_index(lm_model, args.submodule_probe)
    residual_acts_1_list = []
    residual_acts_2_list = []

    def _capture_residual_1(_module, _input, _output):
        if isinstance(_output, tuple):
            _output = _output[0]
        # Sum over sequence dimension: (batch, seq, hidden) -> (batch, hidden)
        residual_acts_1_list.append(_output.sum(dim=1).detach().cpu())

    def _capture_residual_2(_module, _input, _output):
        if isinstance(_output, tuple):
            _output = _output[0]
        # Sum over sequence dimension: (batch, seq, hidden) -> (batch, hidden)
        residual_acts_2_list.append(_output.sum(dim=1).detach().cpu())

    # Extract residual activations for first sentences
    print("Extracting residual activations for first sentences...")
    hook_1 = submodule_probe.register_forward_hook(_capture_residual_1)

    with t.no_grad():
        for i in tqdm(
            range(0, len(sentences_1_train), args.batch_size),
            desc="Extracting residual activations (s1)",
        ):
            batch_1 = sentences_1_train[i : i + args.batch_size]
            batch_tokenized_1 = tokenizer(
                batch_1, return_tensors="pt", padding=True
            ).to(lm_model.device)
            _ = lm_model(**batch_tokenized_1)

    hook_1.remove()
    residual_acts_1 = t.cat(residual_acts_1_list, dim=0).cpu()

    # Extract residual activations for second sentences
    print("Extracting residual activations for second sentences...")
    hook_2 = submodule_probe.register_forward_hook(_capture_residual_2)

    with t.no_grad():
        for i in tqdm(
            range(0, len(sentences_2_train), args.batch_size),
            desc="Extracting residual activations (s2)",
        ):
            batch_2 = sentences_2_train[i : i + args.batch_size]
            batch_tokenized_2 = tokenizer(
                batch_2, return_tensors="pt", padding=True
            ).to(lm_model.device)
            _ = lm_model(**batch_tokenized_2)

    hook_2.remove()
    residual_acts_2 = t.cat(residual_acts_2_list, dim=0).cpu()

    # Compute delta residual activations for difference-based probing
    residual_acts_train = (residual_acts_2 - residual_acts_1).numpy()
    print(f"Delta residual stream activations shape: {residual_acts_train.shape}")

    # Train initial probes on DELTA SAE ACTIVATIONS for feature ranking
    # This ensures probe coefficients match SAE feature dimensions
    print("\nTraining initial probes on delta SAE activations for feature ranking...")
    initial_probes = {}
    for concept in labels_train_dict.keys():
        probe = LogisticRegression(
            random_state=args.seed,
            max_iter=1000,
            class_weight="balanced",
            solver="lbfgs",
            C=1.0,
        )
        probe.fit(sae_train, labels_train_dict[concept])
        initial_probes[concept] = probe
        train_acc = probe.score(sae_train, labels_train_dict[concept])
        test_acc = probe.score(sae_test, labels_test_dict[concept])
        print(
            f"  {concept}: Train acc = {train_acc:.4f}, Test acc = {test_acc:.4f}"
        )

    # Get LR-based feature scores for each concept (using trained probe coefficients)
    print(f"\nExtracting LR feature scores from trained probes...")
    all_scores_dict = {}

    for concept in labels_train_dict.keys():
        print(f"  {concept}...")
        probe = initial_probes[concept]
        # For binary classification, coef_ has shape (1, n_features)
        # Use absolute values as feature importance scores
        all_scores = np.abs(probe.coef_[0])
        all_scores_dict[concept] = all_scores

    # Sweep k values
    print(f"\nSweeping k values: {args.k_values}")
    mcc_method = "union" if args.mcc_union_features else "per-concept"
    print(f"MCC computation method: {mcc_method}")
    results = sweep_k_values_for_plots(
        sae_activations_train=sae_train,
        sae_activations_test=sae_test,
        labels_train=labels_train_dict,
        labels_test=labels_test_dict,
        all_feature_scores=all_scores_dict,
        k_values=args.k_values,
        seed=args.seed,
        verbose=True,
        compute_activation_mcc=True,
        mcc_union_features=args.mcc_union_features,
    )

    # Print results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    for k_idx, k in enumerate(args.k_values):
        print(f"\nk = {k}:")
        for concept in labels_train_dict.keys():
            probe_corr = results[concept]["probe_correlation"][k_idx]
            test_acc = results[concept]["test_acc"][k_idx]
            mcc = results[concept]["mcc"][k_idx]
            print(f"  {concept}:")
            print(f"    Probe Correlation: {probe_corr:.4f}")
            print(f"    Test Acc: {test_acc:.4f}")
            print(f"    MCC: {mcc:.4f}")
        print(
            f"  Aggregate Activation MCC: {results['aggregate_activation_mcc'][k_idx]:.4f}"
        )

    # Create plots
    if args.plot_output or len(labels_train_dict) <= 3:
        print(f"\nCreating plots...")
        num_concepts = len(labels_train_dict)
        fig, axes = plt.subplots(
            1,
            min(num_concepts + 1, 4),
            figsize=(5 * min(num_concepts + 1, 4), 4),
        )

        if num_concepts == 1:
            axes = [axes]

        # Plot each concept's probe correlation
        for idx, concept in enumerate(labels_train_dict.keys()):
            if idx >= 3:  # Max 3 concept plots
                break
            ax = axes[idx] if num_concepts > 1 else axes[0]
            ax.plot(
                results["k_values"],
                results[concept]["probe_correlation"],
                marker="o",
                linewidth=2,
                label="Probe Correlation",
            )
            ax.set_xlabel("k", fontsize=12)
            ax.set_ylabel("Correlation", fontsize=12)
            ax.set_title(f"{concept}", fontsize=14)
            ax.grid(True, alpha=0.3)
            ax.legend()

        # Plot aggregate MCC
        ax_mcc = axes[min(num_concepts, 3)] if num_concepts > 1 else axes[0]
        ax_mcc.plot(
            results["k_values"],
            results["aggregate_activation_mcc"],
            marker="o",
            linewidth=2,
            label="Aggregate MCC",
            color="green",
        )
        ax_mcc.set_xlabel("k", fontsize=12)
        ax_mcc.set_ylabel("MCC", fontsize=12)
        ax_mcc.set_title("MCC (Activation-Label)", fontsize=14)
        ax_mcc.grid(True, alpha=0.3)
        ax_mcc.legend()

        plt.tight_layout()

        if args.plot_output:
            plt.savefig(args.plot_output, dpi=300, bbox_inches="tight")
            print(f"Plots saved to {args.plot_output}")
        else:
            plt.show()

    # Save results to JSON
    if args.output:
        save_results = {
            "k_values": results["k_values"].tolist(),
            "concepts": {},
        }
        for concept in labels_train_dict.keys():
            save_results["concepts"][concept] = {
                "probe_correlation": results[concept][
                    "probe_correlation"
                ].tolist(),
                "train_acc": results[concept]["train_acc"].tolist(),
                "test_acc": results[concept]["test_acc"].tolist(),
                "mcc": results[concept]["mcc"].tolist(),
            }
        save_results["aggregate_activation_mcc"] = results[
            "aggregate_activation_mcc"
        ].tolist()

        with open(args.output, "w") as f:
            json.dump(save_results, f, indent=2)
        print(f"\nResults saved to {args.output}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate SSAE on individual sentence labels"
    )
    parser.add_argument(
        "model_path", type=Path, help="Path to trained model directory"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.1,
        help="Activation threshold for binarization",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=["accuracy", "macrof1"],
        choices=["accuracy", "macrof1", "mcc", "gradient"],
        help="Metrics to compute",
    )
    parser.add_argument("--output", type=Path, help="Output file for results")
    parser.add_argument(
        "--save-probe",
        action="store_true",
        help="Save trained probes to disk",
    )
    parser.add_argument(
        "--probe-output-dir",
        type=Path,
        default=Path("outputs/probes"),
        help="Directory to save trained probes (default: outputs/probes)",
    )
    parser.add_argument(
        "--probe-type",
        type=str,
        choices=["binary", "multiclass"],
        default="binary",
        help="Type of probe to train: binary or multiclass",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for probe training",
    )
    parser.add_argument(
        "--sparse",
        "-s",
        type=str,
        default=None,
        choices=["max_mean_diff", "lr"],
        help="Sparse probing method: max_mean_diff or lr",
    )
    parser.add_argument(
        "--k",
        "-k",
        type=int,
        default=10,
        help="Number of top features to use when --sparse is set",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Batch size for embedding extraction (default: 128)",
    )
    parser.add_argument(
        "--sweep-k-attribution",
        action="store_true",
        help="Perform k-sweep using LR coefficient-based feature selection (method='lr' from heuristic_feature_ranking_binary). "
             "Ranks features by |probe.coef_[0]| - much faster than gradient attribution.",
    )
    parser.add_argument(
        "--k-values",
        nargs="+",
        type=int,
        default=[1, 2, 5, 10, 50],
        help="K values to sweep over (default: 1 2 5 10 50)",
    )
    parser.add_argument(
        "--concepts",
        nargs="+",
        type=str,
        default=None,
        help="Concepts to evaluate (e.g., domain-science sentiment-positive). If not specified, uses all available.",
    )
    parser.add_argument(
        "--sae-activations-path",
        type=Path,
        default=None,
        help="Path to pre-computed SAE activations file (.npy or .pt). If provided, skips activation extraction.",
    )
    parser.add_argument(
        "--lm-model-name",
        type=str,
        default=None,
        help="Language model name (only needed for loading model to extract residual stream activations). If not provided, automatically inferred from SAE config.",
    )
    parser.add_argument(
        "--submodule-steer",
        type=str,
        default=None,
        help="Submodule name for SAE steering (e.g., gpt_neox.layers.5). If not provided, automatically inferred from SAE config.",
    )
    parser.add_argument(
        "--submodule-probe",
        type=str,
        default=None,
        help="Submodule name for probe (e.g., gpt_neox.layers.5). If not provided, defaults to same as --submodule-steer.",
    )
    parser.add_argument(
        "--use-sparsemax",
        action="store_true",
        help="Whether SAE uses sparsemax activation",
    )
    parser.add_argument(
        "--plot-output",
        type=Path,
        default=None,
        help="Path to save the k-sweep plots (e.g., mcc_vs_k.png)",
    )
    parser.add_argument(
        "--causal-intervention",
        action="store_true",
        help="Run causal intervention experiment (requires --sweep-k-attribution)",
    )
    parser.add_argument(
        "--intervention-type",
        type=str,
        choices=["zero", "amplify", "ablate", "add_decoder"],
        default="add_decoder",
        help="Type of intervention to perform (default: add_decoder)",
    )
    parser.add_argument(
        "--intervention-strength",
        type=float,
        default=2.0,
        help="Strength for amplify/add_decoder intervention (default: 2.0)",
    )
    parser.add_argument(
        "--intervention-output",
        type=Path,
        default=None,
        help="Path to save causal intervention matrix plot (e.g., intervention_matrix.png)",
    )
    parser.add_argument(
        "--steering-curves",
        type=str,
        default=None,
        help="Generate steering strength curves for concept pairs (format: 'concept1,concept2;concept3,concept4')",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Limit number of samples for faster processing (useful for large models like Gemma)",
    )
    parser.add_argument(
        "--mcc-union-features",
        action="store_true",
        help="Use union of top-k features across concepts for aggregate MCC (default: per-concept average)",
    )

    args = parser.parse_args()

    # Check if causal intervention experiment is requested
    if args.causal_intervention:
        run_causal_intervention_experiment(args)
        return

    # Check if k-sweep with attribution is requested
    if args.sweep_k_attribution:
        run_k_sweep_attribution(args)
        return

    # Standard evaluation
    results = evaluate_sentence_labels(
        args.model_path, args.threshold, args.metrics, args.batch_size, args.k
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

            # Group scores by prefix (before hyphen) for macro-averaging
            prefix_groups = {}
            for i, label in enumerate(list(top_features.keys())):
                prefix = label.split('-')[0] if '-' in label else label
                if prefix not in prefix_groups:
                    prefix_groups[prefix] = []
                prefix_groups[prefix].append(top_scores[i].item())

            # Compute average for each prefix group
            prefix_averages = {}
            for prefix, group_scores in prefix_groups.items():
                prefix_averages[prefix] = sum(group_scores) / len(group_scores)

            # Compute macro-average across prefix groups
            mcc = sum(prefix_averages.values()) / len(prefix_averages)

            print("\nMax Correlation (MCC) for each concept:")
            print("-" * 50)
            for i, label in enumerate(list(top_features.keys())):
                print(
                    f"  {label}: {top_scores[i]:.4f} (feature {top_features[label]})"
                )

            print("\nPrefix Group Averages:")
            print("-" * 50)
            for prefix, avg in sorted(prefix_averages.items()):
                print(f"  {prefix}: {avg:.4f}")

            print(f"\nMacro-Average MCC (across {len(prefix_groups)} groups): {mcc:.4f}")
            print(
                f"Correlation matrix shape: {data['correlation_matrix'].shape}"
            )

            # Print probe MCC if k-sparse probing was used
            if "probe_mcc_k" in data and data["probe_mcc_k"] is not None:
                print(f"\nProbe MCC (k={data['k']} features):")
                print("-" * 50)
                probe_scores = data["probe_mcc_k"]
                for label, score in probe_scores.items():
                    print(f"  {label}: {score:.4f}")
                avg_probe_mcc = sum(probe_scores.values()) / len(probe_scores)
                print(f"\nAverage Probe MCC: {avg_probe_mcc:.4f}")
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
