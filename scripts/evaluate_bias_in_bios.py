#!/usr/bin/env python3
"""
Simple bias-in-bios gender detection evaluation using MCC.

Loads test samples from bias-in-bios dataset, extracts embeddings,
gets SAE activations, finds max correlation feature, computes MCC.
"""

import torch
import numpy as np
import argparse
import sys
import os
from pathlib import Path
from datasets import load_dataset
from sklearn.metrics import matthews_corrcoef

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ssae import DictLinearAE
from loaders import load_llamascope_checkpoint, load_gemmascope_checkpoint, load_pythia_sae_checkpoint


def load_sae_model(model_path: Path):
    """Load custom SSAE model."""
    import yaml

    # Load config
    config_path = model_path / "cfg.yaml"
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    # Load weights
    weights_path = model_path / "weights.pth"
    state_dict = torch.load(weights_path, map_location="cpu")

    rep_dim = state_dict["encoder.weight"].shape[1]
    hid_dim = state_dict["encoder.weight"].shape[0]

    model = DictLinearAE(rep_dim, hid_dim, cfg.get("norm", "ln"))
    model.load_state_dict(state_dict)
    model.eval()

    return model


def load_reference_sae(embedding_model: str, layer: int = None, hf_token: str = None):
    """Load reference SAE based on embedding model."""
    if embedding_model == "pythia":
        layer = layer or 5
        decoder_weight, decoder_bias, encoder_weight, encoder_bias = load_pythia_sae_checkpoint(layer, hf_token)
    elif embedding_model == "gemma":
        decoder_weight, decoder_bias, encoder_weight, encoder_bias = load_gemmascope_checkpoint()
    elif embedding_model == "llama":
        decoder_weight, decoder_bias, encoder_weight, encoder_bias = load_llamascope_checkpoint()
    else:
        raise ValueError(f"Unknown embedding model: {embedding_model}")

    rep_dim = encoder_weight.shape[1]
    hid_dim = encoder_weight.shape[0]

    model = DictLinearAE(rep_dim, hid_dim, "none")
    model.encoder.weight.data = encoder_weight
    model.encoder.bias.data = encoder_bias
    model.decoder.weight.data = decoder_weight
    model.decoder.bias.data = decoder_bias
    model.eval()

    return model


def extract_embeddings(texts, embedding_model: str, max_samples: int = None):
    """Extract embeddings from texts using specified model."""
    if max_samples:
        texts = texts[:max_samples]

    if embedding_model == "pythia":
        from transformers import AutoTokenizer, AutoModelForCausalLM
        model_name = "EleutherAI/pythia-70m-deduped"
        layer = 5
        batch_size = 32
    elif embedding_model == "gemma":
        from transformers import AutoTokenizer, AutoModelForCausalLM
        model_name = "google/gemma-2-2b-it"
        layer = 16
        batch_size = 16
    else:
        raise ValueError(f"Unknown embedding model: {embedding_model}")

    print(f"Loading {embedding_model} model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    embeddings = []
    print(f"Extracting embeddings for {len(texts)} samples...")

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]

        # Tokenize
        inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=512)

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            # Get embeddings from specified layer, last token
            hidden_states = outputs.hidden_states[layer]  # [batch, seq_len, hidden_dim]

            batch_embeddings = []
            for j, input_ids in enumerate(inputs['input_ids']):
                # Find last non-padding token
                non_pad_indices = (input_ids != tokenizer.pad_token_id).nonzero(as_tuple=True)[0]
                if len(non_pad_indices) > 0:
                    last_token_idx = non_pad_indices[-1]
                    embedding = hidden_states[j, last_token_idx]
                else:
                    embedding = hidden_states[j, 0]  # fallback
                batch_embeddings.append(embedding)

            embeddings.extend(batch_embeddings)

    return torch.stack(embeddings)


def get_sae_activations(model, embeddings):
    """Get SAE activations from embeddings."""
    activations = []
    batch_size = 256

    with torch.no_grad():
        for i in range(0, len(embeddings), batch_size):
            batch = embeddings[i:i+batch_size]
            _, acts = model(batch)
            activations.append(acts.cpu())

    return torch.cat(activations, dim=0)


def compute_max_correlation_mcc(activations, labels, threshold: float = 0.1):
    """Find max correlation feature and compute MCC."""
    activations = activations.float()
    labels = torch.tensor(labels, dtype=torch.float32)

    # Binarize activations
    binary_acts = (activations > threshold).float()

    # Compute correlations
    acts_centered = binary_acts - binary_acts.mean(dim=0, keepdim=True)
    acts_std = acts_centered.norm(dim=0, keepdim=True) + 1e-8

    labels_centered = labels - labels.mean()
    labels_std = labels_centered.norm() + 1e-8

    numerator = acts_centered.T @ labels_centered.unsqueeze(1)
    denominator = acts_std.T * labels_std

    mask = denominator.squeeze() != 0
    correlations = torch.zeros(activations.shape[1])
    correlations[mask] = numerator.squeeze()[mask] / denominator.squeeze()[mask]

    # Find max correlation feature
    abs_correlations = correlations.abs()
    best_feature_idx = abs_correlations.argmax().item()
    best_correlation = correlations[best_feature_idx].item()

    # Get MCC for best feature
    best_activations = binary_acts[:, best_feature_idx].numpy()
    labels_np = labels.numpy()

    mcc = matthews_corrcoef(labels_np, best_activations)

    return {
        "mcc": mcc,
        "feature_idx": best_feature_idx,
        "correlation": best_correlation,
        "max_abs_correlation": abs_correlations.max().item(),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate bias-in-bios gender detection MCC")
    parser.add_argument("model_path", type=Path, help="Path to SSAE model directory")
    parser.add_argument("--embedding-model", choices=["pythia", "gemma", "llama"], default="pythia",
                        help="Embedding model to use")
    parser.add_argument("--compare-sae", action="store_true",
                        help="Compare with reference SAE")
    parser.add_argument("--max-samples", type=int, default=1000,
                        help="Max samples to evaluate")
    parser.add_argument("--threshold", type=float, default=0.1,
                        help="Activation threshold")
    parser.add_argument("--layer", type=int, help="Layer for reference SAE (pythia only)")
    parser.add_argument("--hf-token", type=str, help="Hugging Face token")

    args = parser.parse_args()

    # Load bias-in-bios dataset
    print("Loading bias-in-bios dataset...")
    dataset = load_dataset("LabHC/bias_in_bios")
    test_data = dataset["test"]

    # Sample if needed
    if args.max_samples and len(test_data) > args.max_samples:
        import random
        random.seed(42)
        indices = random.sample(range(len(test_data)), args.max_samples)
        test_data = test_data.select(indices)

    # Extract texts and labels
    texts = [item["hard_text"] for item in test_data]
    gender_labels = [item["gender"] for item in test_data]  # 0=male, 1=female

    print(f"Loaded {len(texts)} samples")
    print(f"Gender distribution: {sum(gender_labels)} female, {len(gender_labels) - sum(gender_labels)} male")

    # Extract embeddings
    embeddings = extract_embeddings(texts, args.embedding_model, args.max_samples)
    print(f"Embeddings shape: {embeddings.shape}")

    # Evaluate custom SSAE
    print(f"\n{'='*50}")
    print("CUSTOM SSAE EVALUATION")
    print(f"{'='*50}")

    custom_model = load_sae_model(args.model_path)
    custom_activations = get_sae_activations(custom_model, embeddings)
    custom_results = compute_max_correlation_mcc(custom_activations, gender_labels, args.threshold)

    print(f"Custom SSAE MCC: {custom_results['mcc']:.4f}")
    print(f"Best feature: {custom_results['feature_idx']}")
    print(f"Correlation: {custom_results['correlation']:.4f}")

    # Compare with reference SAE if requested
    if args.compare_sae:
        print(f"\n{'='*50}")
        print(f"REFERENCE {args.embedding_model.upper()} SAE EVALUATION")
        print(f"{'='*50}")

        try:
            ref_model = load_reference_sae(args.embedding_model, args.layer, args.hf_token)
            ref_activations = get_sae_activations(ref_model, embeddings)
            ref_results = compute_max_correlation_mcc(ref_activations, gender_labels, args.threshold)

            print(f"Reference SAE MCC: {ref_results['mcc']:.4f}")
            print(f"Best feature: {ref_results['feature_idx']}")
            print(f"Correlation: {ref_results['correlation']:.4f}")

            # Final comparison
            print(f"\n{'='*50}")
            print("COMPARISON")
            print(f"{'='*50}")
            print(f"Custom SSAE MCC:    {custom_results['mcc']:.4f}")
            print(f"Reference SAE MCC:  {ref_results['mcc']:.4f}")
            print(f"Difference:         {abs(custom_results['mcc'] - ref_results['mcc']):.4f}")

            if custom_results['mcc'] > ref_results['mcc']:
                print(f"🥇 WINNER: Custom SSAE (+{custom_results['mcc'] - ref_results['mcc']:.4f})")
            elif ref_results['mcc'] > custom_results['mcc']:
                print(f"🥇 WINNER: Reference SAE (+{ref_results['mcc'] - custom_results['mcc']:.4f})")
            else:
                print("🤝 TIE!")

        except Exception as e:
            print(f"❌ Reference SAE Error: {e}")

    print("\n✅ Evaluation complete!")


if __name__ == "__main__":
    main()