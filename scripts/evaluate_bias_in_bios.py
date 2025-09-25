#!/usr/bin/env python3
"""
Evaluate SSAE performance on bias-in-bios dataset using MCC.

This script evaluates trained SSAE models on the bias-in-bios dataset
by computing Matthews Correlation Coefficient (MCC) between SSAE activations
and gender labels binarized on the test split.
"""

import torch as t
import h5py
import json
import argparse
import yaml
import os
import sys
from pathlib import Path
from typing import Dict, Any
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score

# Add parent directory to path to import modules
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from utils.data_utils import load_biasinbios


def heuristic_feature_ranking_binary(
    train_activations: t.Tensor,
    train_labels: np.ndarray,
    method: str = "correlation",
) -> t.Tensor:
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
    acts_centered = train_activations - train_activations.mean(
        dim=0, keepdim=True
    )
    acts_std = acts_centered.norm(dim=0, keepdim=True) + 1e-8

    labels_centered = train_labels_tensor - train_labels_tensor.mean()
    labels_std = labels_centered.norm() + 1e-8

    # Compute correlations for each feature
    numerator = acts_centered.T @ labels_centered.unsqueeze(
        1
    )  # [num_features, 1]
    denominator = acts_std.T * labels_std  # [num_features, 1]

    # Prevent division by zero
    mask = denominator.squeeze() != 0
    correlations = t.zeros(train_activations.shape[1])
    correlations[mask] = (
        numerator.squeeze()[mask] / denominator.squeeze()[mask]
    )

    # Sort by absolute correlation (ascending order, so top features are at the end)
    abs_correlations = correlations.abs()
    sorted_indices = t.argsort(abs_correlations)

    return sorted_indices


def train_probe_on_top_features(
    train_activations: t.Tensor,
    train_labels: np.ndarray,
    sorted_neurons: t.Tensor,
    k: int = 10,
    seed: int = 42,
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
        solver="lbfgs" if k <= 1000 else "saga",
    )

    classifier.fit(train_features, train_labels)

    return classifier


def evaluate_probe_on_top_features(
    classifier: LogisticRegression,
    activations: t.Tensor,
    labels: np.ndarray,
    sorted_neurons: t.Tensor,
    k: int = 10,
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


def train_and_evaluate_probe(
    train_activations: t.Tensor,
    train_labels: np.ndarray,
    test_activations: t.Tensor,
    test_labels: np.ndarray,
    seed: int = 42,
    sparse: str = None,
    k: int = 10,
) -> Dict[str, Any]:
    """
    Train and evaluate a logistic regression probe.

    Args:
        train_activations: Training activations [N_train, num_features]
        train_labels: Training labels [N_train]
        test_activations: Test activations [N_test, num_features]
        test_labels: Test labels [N_test]
        seed: Random seed
        sparse: Feature selection method ("correlation" or None for all features)
        k: Number of top features to use if sparse is specified

    Returns:
        Dictionary with probe results
    """
    print("Training logistic regression model...")

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

        results = {
            "classifier": classifier,
            "top_neurons": top_neurons.tolist(),
            "train_accuracy": train_accuracy,
            "test_accuracy": test_accuracy,
            "method": "sparse",
            "k": k,
        }

    else:
        # Use all features
        classifier = LogisticRegression(
            random_state=seed,
            max_iter=1000,
            class_weight="balanced",
            solver="lbfgs" if train_activations.shape[1] <= 1000 else "saga",
        )

        train_features = train_activations.numpy()
        test_features = test_activations.numpy()

        classifier.fit(train_features, train_labels)

        train_accuracy = classifier.score(train_features, train_labels)
        test_accuracy = classifier.score(test_features, test_labels)

        results = {
            "classifier": classifier,
            "train_accuracy": train_accuracy,
            "test_accuracy": test_accuracy,
            "method": "all_features",
        }

    print(f"Train Accuracy: {results['train_accuracy']:.4f}")
    print(f"Test Accuracy: {results['test_accuracy']:.4f}")

    return results


def load_ssae_model(model_path: Path):
    """Load trained SSAE model."""
    from ssae import DictLinearAE

    # Load model config
    config_path = model_path / "cfg.yaml"
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    # Load model weights
    weights_path = model_path / "weights.pth"
    state_dict = t.load(weights_path, map_location="cpu")

    # Get dimensions from the saved weights
    rep_dim = state_dict["encoder.weight"].shape[1]  # input dimension
    hid_dim = state_dict["encoder.weight"].shape[0]  # hidden dimension

    # Create model with layer norm (standard for bias-in-bios)
    model = DictLinearAE(rep_dim, hid_dim, cfg.get("norm", "ln"))
    model.load_state_dict(state_dict)
    model.eval()

    return model


def load_pythia_sae_model(layer: int = 5):
    """Load Pythia SAE model from Hugging Face hub."""
    from ssae import DictLinearAE
    from loaders import load_pythia_sae_checkpoint

    # Load checkpoint from hub
    decoder_weight, decoder_bias, encoder_weight, encoder_bias = load_pythia_sae_checkpoint(layer)

    rep_dim = encoder_weight.shape[1]  # input dimension
    hid_dim = encoder_weight.shape[0]  # hidden dimension

    print(f"Pythia SAE dimensions: input={rep_dim}, hidden={hid_dim}")

    # Create model with ReLU activation (standard for Pythia SAE)
    model = DictLinearAE(rep_dim, hid_dim, "none")  # No normalization for standard SAE

    # Set weights
    model.encoder.weight.data = encoder_weight
    model.encoder.bias.data = encoder_bias
    model.decoder.weight.data = decoder_weight
    model.decoder.bias.data = decoder_bias

    model.eval()
    return model


def get_bias_in_bios_embeddings_and_labels(
    model_name="google/gemma-2-2b-it",
    layer=16,
    batch_size=64,
    max_samples=None,
):
    """
    Extract embeddings and gender labels for bias-in-bios test data.

    Args:
        model_name: Model to use for embedding extraction
        layer: Layer to extract embeddings from
        batch_size: Batch size for processing
        max_samples: Maximum number of samples to use (None for all)

    Returns:
        tuple: (embeddings, gender_labels) where embeddings are [N, rep_dim]
               and gender_labels are [N] with 0=male, 1=female
    """
    from datasets import load_dataset
    from ssae.store_embeddings import (
        load_model_and_tokenizer,
        extract_embeddings,
    )

    print("Loading bias_in_bios dataset...")
    ds = load_dataset("LabHC/bias_in_bios")

    # Use test split for evaluation
    test_data = ds["test"]
    print(f"Test split has {len(test_data)} samples")

    # Sample if requested
    if max_samples is not None and len(test_data) > max_samples:
        import random

        random.seed(42)
        indices = random.sample(range(len(test_data)), max_samples)
        test_data = test_data.select(indices)
        print(f"Sampled {len(test_data)} test samples")

    # Extract biographies and gender labels
    biographies = []
    gender_labels = []

    for item in test_data:
        bio_text = item["hard_text"]
        gender = item["gender"]  # 0 = male, 1 = female

        biographies.append(bio_text)
        gender_labels.append(gender)

    print(f"Extracted {len(biographies)} biographies")
    print(
        f"Gender distribution: {sum(gender_labels)} female, {len(gender_labels) - sum(gender_labels)} male"
    )

    # Load model and tokenizer
    model, tokenizer, _ = load_model_and_tokenizer(model_name)

    # Create fake pairs for extract_embeddings (we only need single embeddings)
    fake_pairs = [(bio, bio) for bio in biographies]

    # Extract embeddings
    print("Extracting embeddings...")
    embeddings = extract_embeddings(
        fake_pairs, model, tokenizer, layer, "last_token", batch_size
    )

    # Take first embedding from each pair (since both are identical)
    embeddings = embeddings[:, 0, :]  # Shape: [N, rep_dim]

    return embeddings, np.array(gender_labels)


def get_ssae_activations(
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


def compute_mcc_for_gender(
    activations: t.Tensor, gender_labels: np.ndarray, threshold: float = 0.1
):
    """
    Compute MCC between SSAE activations and gender labels WITHOUT PROBING.

    Finds the single feature with maximum correlation to gender labels
    and computes MCC using only that feature's binary activations.

    Args:
        activations: SSAE activations [N, num_features]
        gender_labels: Gender labels [N] with 0=male, 1=female
        threshold: Activation threshold for binarization

    Returns:
        dict: Results including MCC for max correlation feature
    """
    print("Computing MCC for gender detection (max correlation feature, no probing)...")

    # Convert to tensors
    activations = activations.float()
    gender_tensor = t.tensor(gender_labels, dtype=t.float32)

    # Binarize activations
    binary_activations = (activations > threshold).float()  # [N, num_features]

    # Center the data for correlation computation
    acts_centered = binary_activations - binary_activations.mean(
        dim=0, keepdim=True
    )
    acts_std = acts_centered.norm(dim=0, keepdim=True) + 1e-8  # Add epsilon

    labels_centered = gender_tensor - gender_tensor.mean()
    labels_std = labels_centered.norm() + 1e-8  # Add epsilon

    # Compute correlations for each feature
    numerator = acts_centered.T @ labels_centered.unsqueeze(
        1
    )  # [num_features, 1]
    denominator = acts_std.T * labels_std  # [num_features, 1]

    # Prevent division by zero
    mask = denominator.squeeze() != 0
    correlations = t.zeros(activations.shape[1])
    correlations[mask] = (
        numerator.squeeze()[mask] / denominator.squeeze()[mask]
    )

    # Get absolute correlations and find best feature
    abs_correlations = correlations.abs()
    best_feature_idx = abs_correlations.argmax().item()
    best_correlation = correlations[best_feature_idx].item()

    # Compute statistics
    mean_abs_correlation = abs_correlations.mean().item()
    max_abs_correlation = abs_correlations.max().item()

    print(f"Max correlation feature: {best_feature_idx} (correlation: {best_correlation:.4f})")
    print(f"Max absolute correlation: {max_abs_correlation:.4f}")
    print(f"Mean absolute correlation: {mean_abs_correlation:.4f}")

    # MCC computation for the max correlation feature ONLY
    best_activations = binary_activations[:, best_feature_idx]

    # Confusion matrix elements for max correlation feature
    tp = ((best_activations == 1) & (gender_tensor == 1)).sum().item()
    tn = ((best_activations == 0) & (gender_tensor == 0)).sum().item()
    fp = ((best_activations == 1) & (gender_tensor == 0)).sum().item()
    fn = ((best_activations == 0) & (gender_tensor == 1)).sum().item()

    # Compute MCC for max correlation feature (NO PROBING)
    mcc_denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc_best = (tp * tn - fp * fn) / (mcc_denominator + 1e-8)

    # Accuracy for max correlation feature
    accuracy = (tp + tn) / len(gender_labels)

    print(f"MCC (max correlation feature): {mcc_best:.4f}")
    print(f"Accuracy (max correlation feature): {accuracy:.4f}")

    return {
        "best_feature_idx": best_feature_idx,
        "best_correlation": best_correlation,
        "max_abs_correlation": max_abs_correlation,
        "mean_abs_correlation": mean_abs_correlation,
        "mcc_best_feature": mcc_best,
        "accuracy_best_feature": accuracy,
        "confusion_matrix": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
        "correlations_all": correlations.tolist(),
    }


def evaluate_bias_in_bios_single_model(
    model_path: Path,
    threshold: float = 0.1,
    embedding_model: str = "gemma",
    max_samples: int = None,
    probe_k: int = 10,
    probe_seed: int = 42,
) -> Dict[str, Any]:
    """Evaluate SSAE on bias-in-bios gender detection."""

    # Choose model and layer based on embedding_model parameter
    if embedding_model == "pythia":
        model_name = "EleutherAI/pythia-70m-deduped"
        layer = 5
        batch_size = 32  # Reduced to avoid OOM
    elif embedding_model == "gemma":
        model_name = "google/gemma-2-2b-it"
        layer = 16
        batch_size = 16  # Reduced to avoid OOM
    else:
        raise ValueError(
            f"Unknown embedding_model: {embedding_model}. Choose 'pythia' or 'gemma'"
        )

    print(f"Using {embedding_model} model: {model_name}")
    print(f"Layer: {layer}, Batch size: {batch_size}")

    # Get embeddings and labels for test data
    print("Extracting test embeddings...")
    embeddings, gender_labels = get_bias_in_bios_embeddings_and_labels(
        model_name=model_name,
        layer=layer,
        batch_size=batch_size,
        max_samples=max_samples,
    )
    print(f"Embeddings shape: {embeddings.shape}")
    print(f"Gender labels shape: {gender_labels.shape}")

    # Load SSAE model
    model = load_ssae_model(model_path)
    print(f"Loaded SSAE model from {model_path}")

    # Get SSAE activations (reduced batch size to avoid OOM)
    print("Getting SSAE activations...")
    activations = get_ssae_activations(model, embeddings, batch_size=256)
    print(f"Activations shape: {activations.shape}")

    # Split data for probing evaluation
    print("Splitting data for probing evaluation...")
    train_acts, test_acts, train_labels, test_labels = train_test_split(
        activations.numpy(),
        gender_labels,
        test_size=0.3,
        random_state=probe_seed,
        stratify=gender_labels,
    )

    # Convert back to tensors
    train_acts = t.tensor(train_acts, dtype=t.float32)
    test_acts = t.tensor(test_acts, dtype=t.float32)

    print(f"Train split: {train_acts.shape[0]} samples")
    print(f"Test split: {test_acts.shape[0]} samples")

    # Compute MCC on full dataset (without probing)
    print("\n" + "=" * 70)
    print("MCC EVALUATION (MAX CORRELATION FEATURE - NO PROBING)")
    print("=" * 70)

    results = compute_mcc_for_gender(activations, gender_labels, threshold)

    print(f"\n🎯 KEY RESULT - MCC (Max Correlation Feature): {results['mcc_best_feature']:.4f}")
    print(f"📊 Feature Index: {results['best_feature_idx']}")
    print(f"📈 Correlation: {results['best_correlation']:.4f}")
    print(f"✅ Accuracy: {results['accuracy_best_feature']:.4f}")

    # Add probing results
    print("\n" + "=" * 50)
    print("PROBING EVALUATION")
    print("=" * 50)

    # Sparse probing with top k features
    print(f"\nSparse Probing (top {probe_k} features):")
    sparse_results = train_and_evaluate_probe(
        train_acts,
        train_labels,
        test_acts,
        test_labels,
        seed=probe_seed,
        sparse="correlation",
        k=probe_k,
    )

    # Dense probing with all features
    print(f"\nDense Probing (all {activations.shape[1]} features):")
    dense_results = train_and_evaluate_probe(
        train_acts,
        train_labels,
        test_acts,
        test_labels,
        seed=probe_seed,
        sparse=None,
    )

    # Add probing results to main results
    results["probing"] = {
        "sparse": {
            "k": probe_k,
            "top_neurons": sparse_results["top_neurons"],
            "train_accuracy": sparse_results["train_accuracy"],
            "test_accuracy": sparse_results["test_accuracy"],
        },
        "dense": {
            "train_accuracy": dense_results["train_accuracy"],
            "test_accuracy": dense_results["test_accuracy"],
        },
        "data_split": {
            "train_samples": len(train_labels),
            "test_samples": len(test_labels),
            "seed": probe_seed,
        },
    }

    return results


def evaluate_bias_in_bios_with_pythia_sae(
    custom_ssae_path: Path,
    pythia_sae_layer: int = 5,
    threshold: float = 0.1,
    max_samples: int = None,
    embedding_model: str = "pythia",
) -> Dict[str, Any]:
    """
    Simple comparison: Custom SSAE vs Pythia SAE.
    Uses the specified embedding model for both evaluations.

    Args:
        custom_ssae_path: Path to trained custom SSAE model
        pythia_sae_layer: Layer number for Pythia SAE (default: 5)
        threshold: Activation threshold for binarization
        max_samples: Max test samples to evaluate
        embedding_model: Which embedding model to use ("pythia" or "gemma")

    Returns:
        Dictionary with MCC results
    """
    print("=" * 60)
    print(f"BIAS-IN-BIOS: CUSTOM SSAE vs PYTHIA SAE MCC COMPARISON")
    print(f"Using {embedding_model.upper()} embeddings for both models")
    print("=" * 60)

    # Choose embedding model and parameters
    if embedding_model == "pythia":
        model_name = "EleutherAI/pythia-70m-deduped"
        layer = 5
        batch_size = 32
    elif embedding_model == "gemma":
        model_name = "google/gemma-2-2b-it"
        layer = 16
        batch_size = 16
    else:
        raise ValueError(f"Unknown embedding_model: {embedding_model}")

    print(f"Loading {embedding_model} embeddings ({model_name}, layer {layer})...")
    embeddings, gender_labels = get_bias_in_bios_embeddings_and_labels(
        model_name=model_name,
        layer=layer,
        batch_size=batch_size,
        max_samples=max_samples,
    )

    results = {}

    # Evaluate custom SSAE
    print(f"\nEvaluating Custom SSAE...")
    try:
        custom_model = load_ssae_model(custom_ssae_path)
        custom_activations = get_ssae_activations(custom_model, embeddings, batch_size=256)  # Reduced batch size
        custom_mcc_results = compute_mcc_for_gender(custom_activations, gender_labels, threshold)
        custom_mcc = custom_mcc_results['mcc_best_feature']
        results["custom_ssae"] = {"mcc": custom_mcc, "details": custom_mcc_results}
        print(f"✅ Custom SSAE MCC: {custom_mcc:.4f}")
    except Exception as e:
        print(f"❌ Custom SSAE Error: {e}")
        results["custom_ssae"] = {"error": str(e)}

    # Evaluate Pythia SAE
    print(f"\nEvaluating Pythia SAE (layer {pythia_sae_layer})...")
    try:
        pythia_model = load_pythia_sae_model(pythia_sae_layer)
        pythia_activations = get_ssae_activations(pythia_model, embeddings, batch_size=256)  # Reduced batch size
        pythia_mcc_results = compute_mcc_for_gender(pythia_activations, gender_labels, threshold)
        pythia_mcc = pythia_mcc_results['mcc_best_feature']
        results["pythia_sae"] = {"mcc": pythia_mcc, "details": pythia_mcc_results}
        print(f"✅ Pythia SAE MCC: {pythia_mcc:.4f}")
    except Exception as e:
        print(f"❌ Pythia SAE Error: {e}")
        results["pythia_sae"] = {"error": str(e)}

    # Print final comparison
    print("\n" + "=" * 60)
    print("🏆 FINAL MCC COMPARISON")
    print("=" * 60)

    if "error" not in results.get("custom_ssae", {}) and "error" not in results.get("pythia_sae", {}):
        custom_mcc = results["custom_ssae"]["mcc"]
        pythia_mcc = results["pythia_sae"]["mcc"]

        print(f"Custom SSAE MCC:  {custom_mcc:.4f}")
        print(f"Pythia SAE MCC:   {pythia_mcc:.4f}")
        print(f"Difference:       {abs(custom_mcc - pythia_mcc):.4f}")

        if custom_mcc > pythia_mcc:
            print(f"🥇 WINNER: Custom SSAE (+{custom_mcc - pythia_mcc:.4f})")
        elif pythia_mcc > custom_mcc:
            print(f"🥇 WINNER: Pythia SAE (+{pythia_mcc - custom_mcc:.4f})")
        else:
            print("🤝 TIE!")
    else:
        print("⚠️  Cannot compare due to errors")

    return results


def evaluate_bias_in_bios_comparison(
    model_path: Path,
    threshold: float = 0.1,
    max_samples: int = None,
    probe_k: int = 10,
    probe_seed: int = 42,
    compare_models: bool = True,
) -> Dict[str, Any]:
    """
    Compare MCC performance across different embedding models.

    Args:
        model_path: Path to trained SSAE model
        threshold: Activation threshold for binarization
        max_samples: Max test samples to evaluate
        probe_k: Number of top features for sparse probing
        probe_seed: Random seed for probing
        compare_models: Whether to compare across embedding models

    Returns:
        Dictionary with comparison results
    """
    print("=" * 70)
    print("BIAS-IN-BIOS MCC COMPARISON ACROSS EMBEDDING MODELS")
    print("=" * 70)

    comparison_results = {}

    # Models to compare
    models_to_compare = ["pythia", "gemma"] if compare_models else ["gemma"]

    for embedding_model in models_to_compare:
        print(f"\n{'='*50}")
        print(f"EVALUATING WITH {embedding_model.upper()} EMBEDDINGS")
        print(f"{'='*50}")

        try:
            results = evaluate_bias_in_bios_single_model(
                model_path,
                threshold,
                embedding_model,
                max_samples,
                probe_k,
                probe_seed,
            )
            comparison_results[embedding_model] = results

        except Exception as e:
            print(f"Error evaluating {embedding_model}: {e}")
            comparison_results[embedding_model] = {"error": str(e)}

    # Print comparison summary
    if compare_models and len(comparison_results) > 1:
        print("\n" + "=" * 70)
        print("MCC COMPARISON SUMMARY")
        print("=" * 70)

        for model_name, results in comparison_results.items():
            if "error" not in results:
                mcc = results["mcc_best_feature"]
                feature_idx = results["best_feature_idx"]
                correlation = results["best_correlation"]
                accuracy = results["accuracy_best_feature"]

                print(f"\n🔬 {model_name.upper()} Embeddings:")
                print(f"   MCC (Max Correlation Feature): {mcc:.4f}")
                print(f"   Best Feature Index: {feature_idx}")
                print(f"   Correlation: {correlation:.4f}")
                print(f"   Accuracy: {accuracy:.4f}")
            else:
                print(f"\n❌ {model_name.upper()}: {results['error']}")

        # Compare MCCs if both models succeeded
        valid_results = {k: v for k, v in comparison_results.items() if "error" not in v}
        if len(valid_results) >= 2:
            mccs = {model: results["mcc_best_feature"] for model, results in valid_results.items()}
            best_model = max(mccs, key=mccs.get)
            worst_model = min(mccs, key=mccs.get)

            print(f"\n🏆 WINNER: {best_model.upper()} (MCC: {mccs[best_model]:.4f})")
            print(f"📊 MCC Difference: {mccs[best_model] - mccs[worst_model]:.4f}")

            # Detailed comparison table
            print(f"\n📋 DETAILED COMPARISON:")
            print(f"{'Model':<10} {'MCC':<8} {'Feature':<8} {'Correlation':<12} {'Accuracy':<10}")
            print("-" * 50)
            for model, results in valid_results.items():
                print(f"{model:<10} {results['mcc_best_feature']:<8.4f} {results['best_feature_idx']:<8} {results['best_correlation']:<12.4f} {results['accuracy_best_feature']:<10.4f}")

    return comparison_results


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate SSAE on bias-in-bios gender detection using MCC"
    )
    parser.add_argument(
        "model_path", type=Path, help="Path to trained SSAE model directory"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.1,
        help="Activation threshold for binarization (default: 0.1)",
    )
    parser.add_argument(
        "--embedding-model",
        choices=["pythia", "gemma"],
        default="gemma",
        help="Model to use for embeddings (default: gemma)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=1000,
        help="Maximum number of test samples to evaluate (default: 1000)",
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
    parser.add_argument("--output", type=Path, help="Output file for results")
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare MCC across different embedding models (Pythia vs Gemma)",
    )
    parser.add_argument(
        "--compare-sae",
        action="store_true",
        help="Compare custom SSAE vs Pythia SAE on Pythia embeddings",
    )
    parser.add_argument(
        "--pythia-sae-layer",
        type=int,
        default=5,
        help="Pythia SAE layer number (default: 5)",
    )

    args = parser.parse_args()

    if not args.model_path.exists():
        print(f"Error: Model path {args.model_path} does not exist")
        sys.exit(1)

    if args.compare_sae:
        print("=" * 70)
        print("BIAS-IN-BIOS SAE COMPARISON MODE")
        print("=" * 70)
        print(f"Model path: {args.model_path}")
        print(f"Pythia SAE layer: {args.pythia_sae_layer}")
        print(f"Embedding model: {args.embedding_model}")
        print(f"Activation threshold: {args.threshold}")
        print(f"Max samples: {args.max_samples}")

        # Evaluate with SAE comparison
        try:
            results = evaluate_bias_in_bios_with_pythia_sae(
                args.model_path,
                args.pythia_sae_layer,
                args.threshold,
                args.max_samples,
                args.embedding_model,
            )
        except Exception as e:
            print(f"Error during SAE comparison evaluation: {e}")
            import traceback

            traceback.print_exc()
            sys.exit(1)

    elif args.compare:
        print("=" * 70)
        print("BIAS-IN-BIOS MCC COMPARISON MODE")
        print("=" * 70)
        print(f"Model path: {args.model_path}")
        print(f"Activation threshold: {args.threshold}")
        print(f"Max samples: {args.max_samples}")
        print(f"Probe k: {args.probe_k}")
        print(f"Probe seed: {args.probe_seed}")

        # Evaluate with comparison
        try:
            results = evaluate_bias_in_bios_comparison(
                args.model_path,
                args.threshold,
                args.max_samples,
                args.probe_k,
                args.probe_seed,
                compare_models=True,
            )
        except Exception as e:
            print(f"Error during comparison evaluation: {e}")
            import traceback

            traceback.print_exc()
            sys.exit(1)

    else:
        print("=" * 70)
        print("BIAS-IN-BIOS GENDER DETECTION EVALUATION")
        print("=" * 70)
        print(f"Model path: {args.model_path}")
        print(f"Embedding model: {args.embedding_model}")
        print(f"Activation threshold: {args.threshold}")
        print(f"Max samples: {args.max_samples}")
        print(f"Probe k: {args.probe_k}")
        print(f"Probe seed: {args.probe_seed}")

        # Evaluate single model
        try:
            results = evaluate_bias_in_bios_single_model(
                args.model_path,
                args.threshold,
                args.embedding_model,
                args.max_samples,
                args.probe_k,
                args.probe_seed,
            )
        except Exception as e:
            print(f"Error during evaluation: {e}")
            import traceback

            traceback.print_exc()
            sys.exit(1)

    # Print final summary (only for single model evaluation)
    if not args.compare and not args.compare_sae:
        print("\n" + "=" * 70)
        print("FINAL SUMMARY")
        print("=" * 70)

        print(f"\n🏆 PRIMARY METRIC - MCC (MAX CORRELATION FEATURE): {results['mcc_best_feature']:.4f}")
        print(f"   Feature Index: {results['best_feature_idx']}")
        print(f"   Correlation: {results['best_correlation']:.4f}")
        print(f"   Accuracy: {results['accuracy_best_feature']:.4f}")

        print(f"\n📊 CORRELATION STATISTICS:")
        print(f"   Max absolute correlation: {results['max_abs_correlation']:.4f}")
        print(f"   Mean absolute correlation: {results['mean_abs_correlation']:.4f}")

        print(f"\n🔢 CONFUSION MATRIX (Max Correlation Feature):")
        cm = results["confusion_matrix"]
        print(f"   True Positives (Female correctly detected):  {cm['tp']}")
        print(f"   True Negatives (Male correctly detected):    {cm['tn']}")
        print(f"   False Positives (Male misclassified):        {cm['fp']}")
        print(f"   False Negatives (Female misclassified):      {cm['fn']}")

        # Print probing results
        if "probing" in results:
            print(f"\n" + "=" * 70)
            print("PROBING RESULTS")
            print("=" * 70)

            probe_results = results["probing"]

            print(
                f"Data split: {probe_results['data_split']['train_samples']} train, {probe_results['data_split']['test_samples']} test"
            )

            # Sparse probing results
            sparse = probe_results["sparse"]
            print(f"\nSparse Probing (top {sparse['k']} features):")
            print(f"  Top neurons: {sparse['top_neurons']}")
            print(f"  Train accuracy: {sparse['train_accuracy']:.4f}")
            print(f"  Test accuracy: {sparse['test_accuracy']:.4f}")

            # Dense probing results
            dense = probe_results["dense"]
            print(f"\nDense Probing (all features):")
            print(f"  Train accuracy: {dense['train_accuracy']:.4f}")
            print(f"  Test accuracy: {dense['test_accuracy']:.4f}")

            print(f"\nProbing Summary:")
            print(f"  Best sparse test accuracy: {sparse['test_accuracy']:.4f}")
            print(f"  Dense test accuracy: {dense['test_accuracy']:.4f}")
            print(
                f"  Sparse vs Dense difference: {sparse['test_accuracy'] - dense['test_accuracy']:.4f}"
            )

    # Save results if output specified
    if args.output:
        if args.compare_sae:
            save_results = {
                "dataset": "bias-in-bios",
                "model_path": str(args.model_path),
                "pythia_sae_layer": args.pythia_sae_layer,
                "embedding_model": args.embedding_model,
                "evaluation_type": "sae_comparison",
                "threshold": args.threshold,
                "max_samples": args.max_samples,
                "results": results,
            }
        elif args.compare:
            save_results = {
                "dataset": "bias-in-bios",
                "model_path": str(args.model_path),
                "evaluation_type": "embedding_comparison",
                "threshold": args.threshold,
                "max_samples": args.max_samples,
                "probe_k": args.probe_k,
                "probe_seed": args.probe_seed,
                "results": results,
            }
        else:
            save_results = {
                "dataset": "bias-in-bios",
                "model_path": str(args.model_path),
                "embedding_model": args.embedding_model,
                "evaluation_type": "single",
                "threshold": args.threshold,
                "max_samples": args.max_samples,
                "probe_k": args.probe_k,
                "probe_seed": args.probe_seed,
                "results": results,
            }

        with open(args.output, "w") as f:
            json.dump(save_results, f, indent=2)
        print(f"\n✅ Results saved to {args.output}")

    print("\n🎯 Evaluation complete!")


if __name__ == "__main__":
    main()
evaluate_bias_in_bios.py
