#!/usr/bin/env python3
"""
Evaluate SSAE on labeled sentences dataset.

This script provides evaluation of SSAE models on individual sentence labels
using accuracy, macro F1, and MCC metrics.
"""

import torch as t
import json
import argparse
from pathlib import Path
from typing import Dict, Any
import yaml
import os
import gc


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
    """Extract embeddings for individual sentences with memory-efficient processing."""
    import torch
    from transformers import GPTNeoXForCausalLM, AutoTokenizer, AutoModelForCausalLM

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Adjust batch size for large models to prevent OOM
    if "gemma" in model_name.lower():
        batch_size = min(batch_size, 4)  # Gemma needs very small batches
        print(f"Using reduced batch size {batch_size} for Gemma model")

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
            model_name,
            low_cpu_mem_usage=True,
            torch_dtype=torch.float16,  # Use half precision to save memory
        ).to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
    else:
        raise NotImplementedError(f"Model {model_name} not supported")

    model.eval()

    # Extract embeddings directly in batches (avoid fake pairs overhead)
    all_embeddings = []
    n_sentences = len(sentences)

    print(f"Extracting embeddings for {n_sentences} sentences in batches of {batch_size}...")

    with torch.no_grad():
        for i in range(0, n_sentences, batch_size):
            if i % max(batch_size * 10, 100) == 0:
                print(f"  Processing {i}/{n_sentences}...")

            batch_sentences = sentences[i:i + batch_size]

            # Tokenize batch
            inputs = tokenizer(
                batch_sentences,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            ).to(device)

            # Forward pass to get hidden states
            outputs = model(**inputs, output_hidden_states=True)

            # Get embeddings from specified layer, last token position
            attention_mask = inputs["attention_mask"]
            seq_lengths = attention_mask.sum(dim=1) - 1  # -1 for 0-indexing

            hidden_states = outputs.hidden_states[layer]  # [batch, seq_len, hidden_dim]

            # Extract last token embedding for each sequence
            batch_embeddings = []
            for j, seq_len in enumerate(seq_lengths):
                emb = hidden_states[j, seq_len, :].cpu().float()  # Convert back to float32
                batch_embeddings.append(emb)

            all_embeddings.extend(batch_embeddings)

            # Clear intermediate tensors
            del outputs, hidden_states, inputs

            # Clear CUDA cache after each batch to prevent fragmentation
            if device.type == "cuda":
                torch.cuda.empty_cache()

    # Clean up model to free memory
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    gc.collect()

    print(f"  Done extracting {len(all_embeddings)} embeddings")
    return torch.stack(all_embeddings)  # Shape: (N, rep_dim)


def get_activations(
    model: t.nn.Module, embeddings: t.Tensor, batch_size: int = 512
) -> t.Tensor:
    """Get SSAE activations for embeddings."""
    model.eval()
    all_activations = []

    with t.no_grad():
        for i in range(0, len(embeddings), batch_size):
            batch = embeddings[i:i + batch_size]
            # Get encoded activations
            activations = model.encode(batch)
            all_activations.append(activations)

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


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate SSAE on labeled sentences dataset"
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
        "--output", type=Path, help="Output file for results (JSON)"
    )

    args = parser.parse_args()

    # Evaluate model
    results = evaluate_sentence_labels(
        args.model_path,
        threshold=args.threshold,
        metrics=args.metrics,
    )

    # Print results
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)

    for metric, metric_results in results.items():
        print(f"\n{metric.upper()}:")
        if "scores" in metric_results:
            for label, score in metric_results["scores"].items():
                top_feat = metric_results["top_features"][label]
                print(f"  {label}: {score:.4f} (top feature: {top_feat})")
        elif "correlation_matrix" in metric_results:
            print(f"  Correlation matrix shape: {metric_results['correlation_matrix'].shape}")
            print(f"  Top features per label:")
            for label, feat in metric_results["top_features"].items():
                print(f"    {label}: dim {feat}")

    # Save results if output specified
    if args.output:
        save_results = {}
        for metric, metric_results in results.items():
            if "scores" in metric_results:
                save_results[metric] = {
                    "scores": {k: float(v) for k, v in metric_results["scores"].items()},
                    "top_features": {k: int(v) for k, v in metric_results["top_features"].items()},
                }
            elif "correlation_matrix" in metric_results:
                save_results[metric] = {
                    "top_features": metric_results["top_features"],
                    "correlation_matrix_shape": list(metric_results["correlation_matrix"].shape),
                }

        with open(args.output, "w") as f:
            json.dump(save_results, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
