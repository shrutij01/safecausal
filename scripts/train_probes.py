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
import pickle
from tqdm import tqdm


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


def score_identification(
    acts, labels, lamda=0.1, metric="accuracy", probe=None, label_name=None
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

    Returns:
        scores and top_features dict, or (corr_matrix, top_features) for mcc
    """
    if metric == "gradient":
        if probe is None or label_name is None:
            raise ValueError(
                "gradient method requires probe and label_name arguments"
            )

        # Compute gradients using the trained probe
        acts_tensor = t.tensor(acts, dtype=t.float32, requires_grad=True)

        # Get probe logits
        if hasattr(probe, "decision_function"):
            logits = t.tensor(
                probe.decision_function(acts_tensor.detach().numpy()),
                dtype=t.float32,
            )
        else:
            # For pytorch probes
            logits = probe(acts_tensor)

        # Sum logits as objective
        objective = logits.sum()
        objective.backward()

        # Get gradients
        grads = acts_tensor.grad.detach()

        # Compute gradient attribution: grad * activation
        grad_attribution = (grads * acts_tensor.detach()).sum(dim=0)  # (F,)

        # Get top feature
        top_score, top_feature = t.topk(grad_attribution.abs(), k=1)

        return {label_name: top_score.item()}, {label_name: top_feature.item()}

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


def load_model(model_path: Path):
    """Load trained SSAE model and return model with config info."""
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

    # Extract model name and layer from config
    model_name = cfg.get("extra", {}).get("model", "EleutherAI/pythia-70m-deduped")
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


def heuristic_feature_ranking_binary(X, y, method="max_mean_diff"):
    """Rank features using heuristic methods for sparse probing."""
    if method == "max_mean_diff":
        pos_class = y == 1
        neg_class = y == 0
        mean_diff = abs(X[pos_class, :].mean(axis=0) - X[neg_class, :].mean(axis=0))
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


def train_probe_on_top_features(X, y, sorted_idxs, k=10, seed=42):
    """Train probe on top-k features."""
    X_sub = X[:, sorted_idxs[-k:]]
    classifier = LogisticRegression(
        random_state=seed,
        max_iter=1000,
        class_weight="balanced",
        solver="newton-cholesky",
    )
    classifier.fit(X_sub, y)
    return classifier


def train_probes_binary(
    train_activations, train_labels, test_activations, test_labels, seed=42, sparse=None, k=10
):
    """Train binary logistic regression probe."""
    if sparse is not None:
        # Sparse probing: select top-k features
        sorted_neurons = heuristic_feature_ranking_binary(
            train_activations, train_labels, method=sparse
        )
        top_neurons = sorted_neurons[-k:]
        classifier = train_probe_on_top_features(
            train_activations, train_labels, sorted_neurons, k=k, seed=seed
        )

        # Evaluate on top features only
        train_acc = classifier.score(train_activations[:, top_neurons], train_labels)
        test_acc = classifier.score(test_activations[:, top_neurons], test_labels)

        print(f"Train Accuracy: {train_acc:.2f}")
        print(f"Test Accuracy: {test_acc:.2f}")

        return classifier, top_neurons
    else:
        # Standard probing: use all features
        classifier = LogisticRegression(
            random_state=seed,
            max_iter=1000,
            class_weight="balanced",
            solver="newton-cholesky",
        )
        classifier.fit(train_activations, train_labels)
        train_accuracy = classifier.score(train_activations, train_labels)
        test_accuracy = classifier.score(test_activations, test_labels)

        print(f"Train Accuracy: {train_accuracy:.2f}")
        print(f"Test Accuracy: {test_accuracy:.2f}")

        return classifier


def train_probes_multiclass(
    train_activations, train_labels, test_activations, test_labels, seed=42, sparse=None, k=10
):
    """Train multiclass logistic regression probe."""
    if sparse is not None:
        # Sparse probing: select top-k features
        sorted_neurons = heuristic_feature_ranking_binary(
            train_activations, train_labels, method=sparse
        )
        top_neurons = sorted_neurons[-k:]

        # Train on top-k features with multiclass solver
        X_sub = train_activations[:, top_neurons]
        classifier = LogisticRegression(
            random_state=seed,
            max_iter=1000,
            class_weight="balanced",
            solver="lbfgs",
            multi_class="multinomial",
        )
        classifier.fit(X_sub, train_labels)

        # Evaluate on top features only
        train_acc = classifier.score(train_activations[:, top_neurons], train_labels)
        test_acc = classifier.score(test_activations[:, top_neurons], test_labels)

        print(f"Train Accuracy: {train_acc:.2f}")
        print(f"Test Accuracy: {test_acc:.2f}")

        return classifier, top_neurons
    else:
        # Standard probing: use all features
        classifier = LogisticRegression(
            random_state=seed,
            max_iter=1000,
            class_weight="balanced",
            solver="lbfgs",
            multi_class="multinomial",
        )
        classifier.fit(train_activations, train_labels)
        train_accuracy = classifier.score(train_activations, train_labels)
        test_accuracy = classifier.score(test_activations, test_labels)

        print(f"Train Accuracy: {train_accuracy:.2f}")
        print(f"Test Accuracy: {test_accuracy:.2f}")

        return classifier


def train_probes(
    train_activations,
    train_labels,
    test_activations,
    test_labels,
    probe_type="binary",
    seed=42,
    sparse=None,
    k=10,
):
    """
    Train probes based on probe type.

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
        Trained classifier (or tuple of (classifier, top_neurons) if sparse)
    """
    if probe_type == "binary":
        return train_probes_binary(
            train_activations,
            train_labels,
            test_activations,
            test_labels,
            seed,
            sparse,
            k,
        )
    elif probe_type == "multiclass":
        return train_probes_multiclass(
            train_activations,
            train_labels,
            test_activations,
            test_labels,
            seed,
            sparse,
            k,
        )
    else:
        raise ValueError(
            f"Invalid probe_type: {probe_type}. Must be 'binary' or 'multiclass'"
        )


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


def measure_correlation(probe, test_activations, test_labels):
    """Compute Pearson correlation between probe logits and ground truth labels."""
    logits = probe.decision_function(test_activations)
    corr_coef = pearsonr(logits, np.array(test_labels))
    return corr_coef


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
        c: f_list_orig[:, top_feature_idxs[c]].max().item()
        for c in steer_list
    }

    # Coefficient distribution logic
    if len(steer_list) <= 1:
        per_feature_coef = coef
    elif len(steer_list) == 2:
        per_feature_coef = coef
    else:
        raise NotImplementedError("Coefficient distribution for > 2 concepts not yet supported.")

    for idx, x in enumerate(x_list):
        x = t.tensor(x, dtype=t.float32).to("cuda")

        # Get original reconstruction and features
        x_hat_org, f_orig = dictionary(x, return_hidden=True)

        # Apply multi-concept intervention
        f_new = f_orig.clone()
        for concept in steer_list:
            f_new[:, top_feature_idxs[concept]] = per_feature_coef * feature_maxes[concept]

        # Decode and compute delta
        x_hat_int = dictionary.decode(f_new)
        x_hat_delta = x_hat_int - x_hat_org
        x_int = x + x_hat_delta
        x_int_list.append(x_int.squeeze())

    return t.stack(x_int_list, dim=0)


def compute_mcc(
    model_path,
    output_dir,
    test_activations,
    test_labels,
    concept_key,
    concept_value,
    layer_num,
    sparse=None,
    verbose=False
):
    """Compute Matthews Correlation Coefficient for probe quality."""
    if sparse is not None:
        top_neurons_path = os.path.join(
            output_dir,
            f"{concept_key}_{concept_value}_top_neurons.pkl",
        )
        with open(top_neurons_path, "rb") as handle:
            top_neurons = pickle.load(handle)
        test_activations = test_activations[:, top_neurons]

    probe_filename = f"{concept_key}_{concept_value}_layer{layer_num}.joblib"
    probe_path = os.path.join(output_dir, probe_filename)

    if not os.path.exists(probe_path):
        if verbose:
            print(f"Probe file not found for {concept_key}_{concept_value}. Skipping.")
        return None

    probe = joblib.load(probe_path)
    corr_coef = measure_correlation(probe, test_activations, test_labels)[0]
    mcc = abs(corr_coef)

    if verbose:
        print(f"{concept_key}_{concept_value}: {corr_coef:.2f}")

    return mcc


def test_all_with_interventions(
    model,
    dictionary,
    train_activations,
    test_activations_dict,
    test_labels_dict,
    top_features,
    sae_acts,
    output_dir,
    verbose=False
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
                    test_concept_str = f"{test_concept_key}-{test_concept_value}"

                    if test_concept_str not in test_activations_dict:
                        continue

                    if verbose:
                        print(f"{train_concept_str} -> {test_concept_str}")

                    # Load probe
                    probe_filename = f"{test_concept_key}_{test_concept_value}.joblib"
                    probe_path = os.path.join(output_dir, probe_filename)

                    if not os.path.exists(probe_path):
                        if verbose:
                            print(f"Probe not found: {probe_path}")
                        continue

                    probe = joblib.load(probe_path)

                    # Get test activations
                    test_activations_org = test_activations_dict[test_concept_str]
                    test_labels = test_labels_dict[test_concept_str]

                    # Apply intervention
                    with t.no_grad():
                        test_activations_int = extract_activations_with_intervention(
                            test_activations_org, sae_acts, dictionary, top_feature
                        )

                    # Compute probe response difference
                    scores_org = probe.decision_function(test_activations_org)
                    scores_int = probe.decision_function(test_activations_int.detach().cpu().numpy())
                    scores_delta = scores_int - scores_org
                    mean_delta = scores_delta.mean()

                    if verbose:
                        print(f"  Delta: {mean_delta:.4f}")

                    score_matrix[IDX[train_concept_str]][IDX[test_concept_str]] = mean_delta

    return score_matrix, IDX


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

    # Load model first to get config
    model = load_model(model_path)
    print(f"Loaded model from {model_path}")
    print(f"Model: {model.model_name}, Layer: {model.layer}, Rep dim: {model.rep_dim}")

    # Get sentence embeddings using model's config
    print("Extracting sentence embeddings...")
    embeddings = get_sentence_embeddings(sentences, model.model_name, model.layer)
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

    args = parser.parse_args()

    # Evaluate model
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

            print("\nMax Correlation (MCC) for each concept:")
            print("-" * 50)
            for i, label in enumerate(list(top_features.keys())):
                print(f"  {label}: {top_scores[i]:.4f} (feature {top_features[label]})")

            print(f"\nAverage MCC: {mcc:.4f}")
            print(f"Correlation matrix shape: {data['correlation_matrix'].shape}")
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
