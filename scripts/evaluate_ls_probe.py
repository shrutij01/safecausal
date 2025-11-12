import torch
import torch as t
import h5py
import json
import argparse
import math
from pathlib import Path
from typing import Dict, Tuple, Any, List
import yaml
import os
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


class Dataset:
    """Simple dataset class to match the reference code interface."""
    def __init__(self, filepath):
        self.examples, self.labels_binary = load_labeled_sentences_test()


def load_dataset(filepath):
    """Load dataset from filepath."""
    return Dataset(filepath)


def score_identification(acts, labels, lamda=0.1, metric="accuracy"):
    scores = {}
    top_features = {}
    label_matrix = t.stack([t.Tensor(labels[l]) for l in labels if "reading" not in l], dim=0)    # N x L
    label_matrix = label_matrix.to(dtype=acts.dtype)

    for label_name in labels:
        label_vec_bool = t.tensor(labels[label_name], dtype=torch.bool)
        if metric == "mcc":
            label_vec = t.Tensor(labels[label_name])   # N
        else:
            label_vec = t.tensor(labels[label_name])

        # Note: Binarizing iVAE latents with a simple threshold might not be optimal.
        # The MCC metric, which uses correlation, is likely more robust.
        feature_labels = acts.T > lamda     # F x N
        if metric == "accuracy":
            matches = (feature_labels == label_vec)
            accuracies = matches.sum(dim=1) / label_vec.shape[-1]
            accuracy = accuracies.max()
            top_features[label_name] = accuracies.argmax()
            scores[label_name] = accuracy
        elif metric == "macrof1":
            true_positives = (feature_labels & label_vec_bool).sum(dim=1).float()
            false_positives = (feature_labels & ~label_vec_bool).sum(dim=1).float()
            false_negatives = (~feature_labels & label_vec_bool).sum(dim=1).float()

            precision = true_positives / (true_positives + false_positives + 1e-10)
            recall = true_positives / (true_positives + false_negatives + 1e-10)

            f1_scores = 2 * precision * recall / (precision + recall + 1e-10)

            top_feature = f1_scores.argmax()
            max_f1 = f1_scores[top_feature]

            top_features[label_name] = top_feature
            scores[label_name] = max_f1
        elif metric == "mcc":
            acts_centered = acts - acts.mean(dim=0, keepdim=True)
            acts_std = acts_centered.norm(dim=0, keepdim=True)
            label_matrix_centered = label_matrix.T - label_matrix.T.mean(dim=0, keepdim=True)
            label_matrix_std = label_matrix_centered.norm(dim=0, keepdim=True)

            numerator = acts_centered.T @ label_matrix_centered
            denominator = acts_std.T * label_matrix_std

            mask = denominator != 0
            corr_matrix = t.zeros_like(numerator)
            corr_matrix[mask] = numerator[mask] / denominator[mask]

            top_feature_indices = corr_matrix.argmax(dim=0)
            ls = [l for l in labels if "reading" not in l]
            top_features = {label_name: top_feature_indices[i].item() for i, label_name in enumerate(list(ls))}
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

    # Load model weights
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
    from transformers import GPTNeoXForCausalLM, AutoTokenizer

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


def extract_sae_top_features(
    train_filepath,
    model,
    embeddings,
    method="corr",
    lamda=0.1,
    batch_size=16,
):
    """
    Extract top SAE features using either correlation or gradient-based methods.

    Args:
        train_filepath: Path to training data
        model: SSAE model
        embeddings: Pre-computed embeddings for the dataset
        method: Feature selection method ("corr" or "gradient")
        lamda: Threshold for binarizing activations
        batch_size: Batch size for processing

    Returns:
        top_features: Dictionary mapping label names to top feature indices
        acts: Activation tensor (num_examples, num_hidden)
    """
    if method not in ("corr", "gradient"):
        raise ValueError("Invalid feature selection method. Pick one of ['corr', 'gradient']")

    dataset = load_dataset(train_filepath)
    num_examples = len(dataset.examples)

    # For now, we only support correlation-based method
    # Gradient method would require probe model which is not implemented yet
    if method == "gradient":
        raise NotImplementedError("Gradient-based method requires probe model, not yet implemented")

    # Get activations in batches
    print("Computing SAE activations...")
    acts = get_activations(model, embeddings, batch_size=batch_size)

    # Compute top features using correlation
    print("Computing feature-label correlations...")
    scores, top_features = score_identification(
        acts,
        dataset.labels_binary,
        lamda=lamda,
        metric="mcc"
    )

    return top_features, acts, scores


def evaluate_ls_probe(
    model_path: Path,
    data_filepath: str = None,
    threshold: float = 0.1,
    method: str = "corr",
    metrics: list = ["mcc"],
) -> Dict[str, Any]:
    """Evaluate SSAE on labeled sentences using probe-based feature identification."""

    # Use default test file if not provided
    if data_filepath is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        data_filepath = os.path.join(
            script_dir,
            "..",
            "data",
            "labeled-sentences",
            "labeled_sentences_large_deduped_test.jsonl",
        )

    # Load test sentences and labels
    dataset = load_dataset(data_filepath)
    print(f"Loaded {len(dataset.examples)} test sentences")
    print(f"Available labels: {list(dataset.labels_binary.keys())}")

    # Get sentence embeddings
    print("Extracting sentence embeddings...")
    embeddings = get_sentence_embeddings(dataset.examples)
    print(f"Embeddings shape: {embeddings.shape}")

    # Load model
    model = load_model(model_path)
    print(f"Loaded model from {model_path}")

    # Extract top features and activations
    print(f"Extracting top features using method: {method}")
    top_features, activations, corr_matrix = extract_sae_top_features(
        data_filepath,
        model,
        embeddings,
        method=method,
        lamda=threshold,
    )

    print(f"Activations shape: {activations.shape}")

    # Prepare results
    results = {}

    # For MCC metric
    if "mcc" in metrics:
        top_scores = corr_matrix.max(dim=0).values
        mcc_scores = {}
        for i, label in enumerate(list(top_features.keys())):
            mcc_scores[label] = float(top_scores[i])

        results["mcc"] = {
            "scores": mcc_scores,
            "average_mcc": float(top_scores.mean().item()),
            "top_features": top_features,
            "correlation_matrix": corr_matrix,
        }

    # For other metrics if requested
    for metric in metrics:
        if metric != "mcc":
            scores, top_feats = score_identification(
                activations, dataset.labels_binary, threshold, metric
            )
            results[metric] = {
                "scores": scores,
                "top_features": top_feats,
            }

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate SSAE on labeled sentences using probe-based feature identification"
    )
    parser.add_argument(
        "model_path", type=Path, help="Path to trained SSAE model directory"
    )
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="Path to test data file (default: labeled_sentences_large_deduped_test.jsonl)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.1,
        help="Activation threshold for binarization",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="corr",
        choices=["corr", "gradient"],
        help="Feature selection method (corr or gradient)",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=["mcc"],
        choices=["accuracy", "macrof1", "mcc"],
        help="Metrics to compute",
    )
    parser.add_argument("--output", type=Path, help="Output file for results")

    args = parser.parse_args()

    # Evaluate model
    results = evaluate_ls_probe(
        args.model_path,
        args.data,
        args.threshold,
        args.method,
        args.metrics,
    )

    # Print results
    print("\n" + "=" * 70)
    print("LABELED SENTENCES PROBE EVALUATION RESULTS")
    print("=" * 70)

    for metric, data in results.items():
        print(f"\n{metric.upper()}:")
        if metric == "mcc":
            scores = data["scores"]
            top_features = data["top_features"]
            avg_mcc = data["average_mcc"]

            print(f"Average MCC: {avg_mcc:.4f}")
            print("\nPer-label results:")
            for label, score in scores.items():
                feature_idx = top_features.get(label, "N/A")
                print(f"  {label}: {score:.4f} (feature {feature_idx})")

        else:
            scores = data["scores"]
            top_features = data["top_features"]
            avg_score = sum(scores.values()) / len(scores) if scores else 0

            print(f"Average {metric}: {avg_score:.4f}")
            print("\nPer-label results:")
            for concept, score in scores.items():
                feature_idx = top_features[concept]
                print(f"  {concept}: {score:.4f} (feature {feature_idx})")

    # Save results if output specified
    if args.output:
        # Convert tensors to lists for JSON serialization
        save_results = {}
        save_results["dataset"] = "labeled-sentences"
        save_results["method"] = args.method
        save_results["threshold"] = args.threshold

        for metric, data in results.items():
            if metric == "mcc":
                save_results[metric] = {
                    "scores": data["scores"],
                    "average_mcc": data["average_mcc"],
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
