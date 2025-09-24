import torch as t
import h5py
import json
import argparse
from pathlib import Path
from typing import Dict, Tuple, Any
import yaml
import os
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def load_jsonl(filepath: str):
    with open(filepath, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def heuristic_feature_ranking_binary(train_activations: t.Tensor, train_labels: np.ndarray, method: str = "correlation") -> t.Tensor:
    """
    Rank features based on their correlation with binary labels.

    Args:
        train_activations: SSAE activations [N, num_features]
        train_labels: Binary labels [N] with 0/1 values
        method: Ranking method ("correlation")

    Returns:
        Tensor of feature indices sorted by correlation (ascending order)
    """
    train_activations = train_activations.float()
    train_labels_tensor = t.tensor(train_labels, dtype=t.float32)

    # Center the data for correlation computation
    acts_centered = train_activations - train_activations.mean(dim=0, keepdim=True)
    acts_std = acts_centered.norm(dim=0, keepdim=True) + 1e-8

    labels_centered = train_labels_tensor - train_labels_tensor.mean()
    labels_std = labels_centered.norm() + 1e-8

    # Compute correlations for each feature
    numerator = acts_centered.T @ labels_centered.unsqueeze(1)  # [num_features, 1]
    denominator = acts_std.T * labels_std  # [num_features, 1]

    # Prevent division by zero
    mask = denominator.squeeze() != 0
    correlations = t.zeros(train_activations.shape[1])
    correlations[mask] = (numerator.squeeze()[mask] / denominator.squeeze()[mask])

    # Sort by absolute correlation (ascending order, so top features are at the end)
    abs_correlations = correlations.abs()
    sorted_indices = t.argsort(abs_correlations)

    return sorted_indices


def train_probe_on_top_features(
    train_activations: t.Tensor,
    train_labels: np.ndarray,
    sorted_neurons: t.Tensor,
    k: int = 10,
    seed: int = 42
) -> LogisticRegression:
    """
    Train a logistic regression probe on top k features.

    Args:
        train_activations: Training activations [N, num_features]
        train_labels: Training labels [N]
        sorted_neurons: Feature indices sorted by importance
        k: Number of top features to use
        seed: Random seed

    Returns:
        Trained LogisticRegression classifier
    """
    # Get top k features (take from the end since sorted in ascending order)
    top_features = sorted_neurons[-k:]

    # Extract activations for top features
    train_features = train_activations[:, top_features].numpy()

    # Train logistic regression
    classifier = LogisticRegression(
        random_state=seed,
        max_iter=1000,
        class_weight="balanced",
        solver="lbfgs" if k <= 1000 else "saga"
    )

    classifier.fit(train_features, train_labels)

    return classifier


def evaluate_probe_on_top_features(
    classifier: LogisticRegression,
    activations: t.Tensor,
    labels: np.ndarray,
    sorted_neurons: t.Tensor,
    k: int = 10
) -> float:
    """
    Evaluate probe on top k features.

    Args:
        classifier: Trained LogisticRegression classifier
        activations: Test activations [N, num_features]
        labels: Test labels [N]
        sorted_neurons: Feature indices sorted by importance
        k: Number of top features to use

    Returns:
        Accuracy score
    """
    # Get top k features
    top_features = sorted_neurons[-k:]

    # Extract activations for top features
    test_features = activations[:, top_features].numpy()

    # Predict and compute accuracy
    predictions = classifier.predict(test_features)
    accuracy = accuracy_score(labels, predictions)

    return accuracy


def train_and_evaluate_probe_for_concept(
    train_activations: t.Tensor,
    train_labels: np.ndarray,
    test_activations: t.Tensor,
    test_labels: np.ndarray,
    concept_name: str,
    seed: int = 42,
    sparse: str = None,
    k: int = 10
) -> Dict[str, Any]:
    """
    Train and evaluate a logistic regression probe for a specific concept.

    Args:
        train_activations: Training activations [N_train, num_features]
        train_labels: Training labels [N_train]
        test_activations: Test activations [N_test, num_features]
        test_labels: Test labels [N_test]
        concept_name: Name of the concept being probed
        seed: Random seed
        sparse: Feature selection method ("correlation" or None for all features)
        k: Number of top features to use if sparse is specified

    Returns:
        Dictionary with probe results
    """
    results = {}

    if sparse is not None:
        # Use sparse probing with top k features
        sorted_neurons = heuristic_feature_ranking_binary(
            train_activations, train_labels, method=sparse
        )
        top_neurons = sorted_neurons[-k:]

        classifier = train_probe_on_top_features(
            train_activations, train_labels, sorted_neurons, k=k, seed=seed
        )

        train_accuracy = evaluate_probe_on_top_features(
            classifier, train_activations, train_labels, sorted_neurons, k=k
        )
        test_accuracy = evaluate_probe_on_top_features(
            classifier, test_activations, test_labels, sorted_neurons, k=k
        )

        # Get probe logits for MCC calculation
        top_features = sorted_neurons[-k:]
        test_features = test_activations[:, top_features].numpy()
        probe_logits = classifier.decision_function(test_features)

        results = {
            "classifier": classifier,
            "top_neurons": top_neurons.tolist(),
            "train_accuracy": train_accuracy,
            "test_accuracy": test_accuracy,
            "probe_logits": probe_logits,
            "method": "sparse",
            "k": k,
            "concept": concept_name
        }

    else:
        # Use all features
        classifier = LogisticRegression(
            random_state=seed,
            max_iter=1000,
            class_weight="balanced",
            solver="lbfgs" if train_activations.shape[1] <= 1000 else "saga"
        )

        train_features = train_activations.numpy()
        test_features = test_activations.numpy()

        classifier.fit(train_features, train_labels)

        train_accuracy = classifier.score(train_features, train_labels)
        test_accuracy = classifier.score(test_features, test_labels)

        # Get probe logits for MCC calculation
        probe_logits = classifier.decision_function(test_features)

        results = {
            "classifier": classifier,
            "train_accuracy": train_accuracy,
            "test_accuracy": test_accuracy,
            "probe_logits": probe_logits,
            "method": "all_features",
            "concept": concept_name
        }

    return results


def compute_mcc_between_logits_and_labels(probe_logits: np.ndarray, labels: np.ndarray) -> float:
    """
    Compute Matthews Correlation Coefficient between probe logits and labels.

    Args:
        probe_logits: Probe decision function outputs (continuous values)
        labels: Binary labels (0/1)

    Returns:
        MCC value
    """
    # Convert to tensors for computation
    logits_tensor = t.tensor(probe_logits, dtype=t.float32)
    labels_tensor = t.tensor(labels, dtype=t.float32)

    # Center the data
    logits_centered = logits_tensor - logits_tensor.mean()
    logits_std = logits_centered.norm() + 1e-8

    labels_centered = labels_tensor - labels_tensor.mean()
    labels_std = labels_centered.norm() + 1e-8

    # Compute correlation (which is MCC for continuous vs binary)
    numerator = logits_centered @ labels_centered
    denominator = logits_std * labels_std

    mcc = numerator / denominator
    return mcc.item()


def load_labeled_sentences_test(max_samples=None):
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

    # Limit samples if specified
    if max_samples is not None and len(labeled_sentences) > max_samples:
        labeled_sentences = labeled_sentences[:max_samples]

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

    from ssae.store_embeddings import (
        extract_embeddings,
        load_model_and_tokenizer,
    )
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Use the existing load_model_and_tokenizer function that supports multiple models
    model, tokenizer, _ = load_model_and_tokenizer(model_name)

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
    embedding_model: str = "pythia",
    max_samples: int = None,
    probe_k: int = 10,
    probe_seed: int = 42,
    target_concepts: list = ["domain-science", "sentiment-positive"]
) -> Dict[str, Any]:
    """Evaluate SSAE on individual sentence labels."""

    # Load test sentences and labels
    sentences, labels = load_labeled_sentences_test(max_samples)
    print(f"Loaded {len(sentences)} test sentences")
    print(f"Available labels: {list(labels.keys())}")

    # Get sentence embeddings
    print("Extracting sentence embeddings...")

    # Choose model based on embedding_model parameter
    if embedding_model == "pythia":
        model_name = "EleutherAI/pythia-70m-deduped"
        layer = 5
    elif embedding_model == "gemma":
        model_name = "google/gemma-2-2b-it"
        layer = 16
    else:
        raise ValueError(
            f"Unknown embedding_model: {embedding_model}. Choose 'pythia' or 'gemma'"
        )

    embeddings = get_sentence_embeddings(
        sentences, model_name=model_name, layer=layer
    )
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

    # Add probing evaluation for target concepts
    print("\n" + "="*60)
    print("PROBING EVALUATION FOR TARGET CONCEPTS")
    print("="*60)

    probing_results = {}

    for concept in target_concepts:
        if concept not in labels:
            print(f"Warning: Concept '{concept}' not found in labels. Skipping.")
            continue

        print(f"\nEvaluating concept: {concept}")

        # Convert concept labels to binary (0/1)
        concept_labels = np.array([int(label) for label in labels[concept]])

        # Check class distribution
        positive_count = np.sum(concept_labels)
        total_count = len(concept_labels)
        print(f"  Label distribution: {positive_count}/{total_count} positive ({positive_count/total_count:.2%})")

        if positive_count == 0 or positive_count == total_count:
            print(f"  Skipping {concept}: all labels are the same class")
            continue

        # Split data for probing evaluation
        train_acts, test_acts, train_labels, test_labels = train_test_split(
            activations.numpy(), concept_labels,
            test_size=0.3, random_state=probe_seed, stratify=concept_labels
        )

        # Convert back to tensors
        train_acts = t.tensor(train_acts, dtype=t.float32)
        test_acts = t.tensor(test_acts, dtype=t.float32)

        print(f"  Train split: {len(train_labels)} samples")
        print(f"  Test split: {len(test_labels)} samples")

        # Sparse probing with varying k values
        sparse_results = {}
        k_values = [5, 10, 15, 20] if probe_k == 10 else [probe_k]  # Test multiple k values or just the specified one

        for k in k_values:
            if k > activations.shape[1]:  # Skip if k is larger than number of features
                continue

            print(f"  Sparse probing with k={k}...")
            sparse_result = train_and_evaluate_probe_for_concept(
                train_acts, train_labels, test_acts, test_labels,
                concept, seed=probe_seed, sparse="correlation", k=k
            )

            # Compute MCC between probe logits and test labels
            mcc = compute_mcc_between_logits_and_labels(sparse_result["probe_logits"], test_labels)
            sparse_result["logits_labels_mcc"] = mcc

            sparse_results[f"k_{k}"] = {
                "top_neurons": sparse_result["top_neurons"],
                "train_accuracy": sparse_result["train_accuracy"],
                "test_accuracy": sparse_result["test_accuracy"],
                "logits_labels_mcc": sparse_result["logits_labels_mcc"],
                "k": k
            }

            print(f"    Train accuracy: {sparse_result['train_accuracy']:.4f}")
            print(f"    Test accuracy: {sparse_result['test_accuracy']:.4f}")
            print(f"    MCC (logits vs labels): {mcc:.4f}")

        # Dense probing with all features
        print(f"  Dense probing (all {activations.shape[1]} features)...")
        dense_result = train_and_evaluate_probe_for_concept(
            train_acts, train_labels, test_acts, test_labels,
            concept, seed=probe_seed, sparse=None
        )

        # Compute MCC between probe logits and test labels
        dense_mcc = compute_mcc_between_logits_and_labels(dense_result["probe_logits"], test_labels)
        dense_result["logits_labels_mcc"] = dense_mcc

        print(f"    Train accuracy: {dense_result['train_accuracy']:.4f}")
        print(f"    Test accuracy: {dense_result['test_accuracy']:.4f}")
        print(f"    MCC (logits vs labels): {dense_mcc:.4f}")

        # Store results for this concept
        probing_results[concept] = {
            "sparse": sparse_results,
            "dense": {
                "train_accuracy": dense_result["train_accuracy"],
                "test_accuracy": dense_result["test_accuracy"],
                "logits_labels_mcc": dense_result["logits_labels_mcc"]
            },
            "data_split": {
                "train_samples": len(train_labels),
                "test_samples": len(test_labels),
                "positive_ratio": positive_count / total_count,
                "seed": probe_seed
            }
        }

    # Add probing results to main results
    results["probing"] = probing_results

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
        choices=["accuracy", "macrof1", "mcc"],
        help="Metrics to compute",
    )
    parser.add_argument(
        "--embedding-model",
        choices=["pythia", "gemma"],
        default="pythia",
        help="Model to use for sentence embeddings (pythia=512D, gemma=2304D)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=100,
        help="Maximum number of test sentences to evaluate (default: 100)",
    )
    parser.add_argument(
        "--probe-k",
        type=int,
        default=10,
        help="Number of top features to use in sparse probing (default: 10)",
    )
    parser.add_argument(
        "--probe-seed",
        type=int,
        default=42,
        help="Random seed for probing train/test split (default: 42)",
    )
    parser.add_argument(
        "--target-concepts",
        nargs="+",
        default=["domain-science", "sentiment-positive"],
        help="Concepts to focus on for probing evaluation",
    )
    parser.add_argument("--output", type=Path, help="Output file for results")

    args = parser.parse_args()

    # Evaluate model
    results = evaluate_sentence_labels(
        args.model_path, args.threshold, args.metrics, args.embedding_model,
        args.max_samples, args.probe_k, args.probe_seed, args.target_concepts
    )

    # Print results
    print("\n" + "=" * 70)
    print("SENTENCE-LEVEL EVALUATION RESULTS")
    print("=" * 70)

    for metric, data in results.items():
        if metric == "probing":
            # Handle probing results separately
            continue

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

    # Print probing results
    if "probing" in results and results["probing"]:
        print(f"\n" + "=" * 70)
        print("PROBING RESULTS")
        print("=" * 70)

        for concept, probe_data in results["probing"].items():
            print(f"\n{concept.upper()}:")

            # Data split info
            data_split = probe_data["data_split"]
            print(f"  Data split: {data_split['train_samples']} train, {data_split['test_samples']} test")
            print(f"  Positive ratio: {data_split['positive_ratio']:.2%}")

            # Sparse probing results
            print(f"\n  Sparse Probing Results:")
            for k_key, sparse_result in probe_data["sparse"].items():
                k = sparse_result["k"]
                print(f"    k={k}:")
                print(f"      Top neurons: {sparse_result['top_neurons']}")
                print(f"      Train accuracy: {sparse_result['train_accuracy']:.4f}")
                print(f"      Test accuracy: {sparse_result['test_accuracy']:.4f}")
                print(f"      MCC (logits vs labels): {sparse_result['logits_labels_mcc']:.4f}")

            # Dense probing results
            dense = probe_data["dense"]
            print(f"\n  Dense Probing Results:")
            print(f"    Train accuracy: {dense['train_accuracy']:.4f}")
            print(f"    Test accuracy: {dense['test_accuracy']:.4f}")
            print(f"    MCC (logits vs labels): {dense['logits_labels_mcc']:.4f}")

            # Summary
            if probe_data["sparse"]:
                best_sparse_k = max(probe_data["sparse"].keys(),
                                  key=lambda k: probe_data["sparse"][k]["logits_labels_mcc"])
                best_sparse = probe_data["sparse"][best_sparse_k]
                print(f"\n  Summary for {concept}:")
                print(f"    Best sparse (k={best_sparse['k']}): MCC={best_sparse['logits_labels_mcc']:.4f}, Acc={best_sparse['test_accuracy']:.4f}")
                print(f"    Dense: MCC={dense['logits_labels_mcc']:.4f}, Acc={dense['test_accuracy']:.4f}")
                print(f"    Improvement: MCC={best_sparse['logits_labels_mcc'] - dense['logits_labels_mcc']:.4f}, Acc={best_sparse['test_accuracy'] - dense['test_accuracy']:.4f}")

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
