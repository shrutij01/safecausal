"""
Evaluate cosine similarities using different methods (decoder directions, PCA).
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import utils.data_utils as utils
from loaders import (
    TestDataLoader,
    load_llamascope_checkpoint,
    load_gemmascope_checkpoint,
    load_ssae_models,
)
from baselines.pca import pca_transform


def get_max_cos_and_steering_vector_for_concept(
    z, z_tilde, decoder_weight, decoder_bias
):
    """
    Compute max cosine similarities and steering vector for a concept.

    Args:
        z: [B, D] tensor of original vectors
        z_tilde: [B, D] tensor of target vectors
        decoder_weight: [V, D] decoder weight matrix
        decoder_bias: [D] decoder bias vector

    Returns:
        tuple: (mean_cos, std_cos, steering_vector)
    """
    z = F.normalize(z, dim=1)  # [B, D]
    z_tilde = F.normalize(z_tilde, dim=1)  # [B, D]
    decoder = decoder_weight.to(z.device)
    decoder_bias = F.normalize(decoder_bias, dim=0).to(z.device)

    B, D = z.shape
    V = decoder.shape[0]

    # z: [B, D], decoder: [V, D]
    z_tilde_hat = (
        z.unsqueeze(2)
        + decoder.unsqueeze(0)
        + decoder_bias.unsqueeze(0).unsqueeze(0)
    )  # [B, D, V]

    z_tilde_hat = F.normalize(z_tilde_hat, dim=1)  # [B, D, V]
    z_tilde_expanded = z_tilde.unsqueeze(2).expand(-1, -1, V)  # [B, D, V]

    # Compute cosine similarities: [B, V]
    cosines = torch.sum(z_tilde_hat * z_tilde_expanded, dim=1)  # [B, V]
    max_cosines = cosines.max(dim=1).values  # [B]
    indices = cosines.argmax(dim=1)  # [B]

    # Get most frequent steering vector
    from collections import Counter

    indices_list = [int(i.item()) for i in indices]
    counter = Counter(indices_list)
    most_frequent_index, count = counter.most_common(1)[0]

    steering_vector = (
        decoder[most_frequent_index]
        if most_frequent_index is not None
        else None
    )

    return max_cosines.mean().item(), max_cosines.std().item(), steering_vector


def take_pca(z_test, tilde_z_test, device):
    """
    Compute cosine similarities using PCA approach.

    Args:
        z_test: Test data tensor
        tilde_z_test: Target test data tensor
        device: Computation device

    Returns:
        list: Cosine similarities for each sample
    """
    shifts = utils.tensorify((tilde_z_test - z_test), device)
    shifts_transformed, components, mean = pca_transform(shifts.float())

    pca_vec = (components.sum(dim=0, keepdim=True) + mean).mean(0)

    z_test_tensor = utils.tensorify(z_test, device)
    z_pca = F.normalize(z_test_tensor) + pca_vec
    z_pca = F.normalize(z_pca)
    z_pca = utils.numpify(z_pca)

    cosines_pca = []
    for i in range(tilde_z_test.shape[0]):
        cosines_pca.append(
            cosine_similarity(
                tilde_z_test[i].reshape(1, -1), z_pca[i].reshape(1, -1)
            )
        )

    return cosines_pca


def detect_model_type_from_config(dataconfig):
    """Detect the appropriate SAE model type from data config."""
    if hasattr(dataconfig, 'model_name'):
        model_name = dataconfig.model_name.lower()
        if 'gemma' in model_name:
            return "gemmascope"
        elif 'llama' in model_name:
            return "llamascope"
        elif 'pythia' in model_name:
            return "ssae"

    # Fallback: check for other config fields
    if hasattr(dataconfig, 'model_id'):
        model_id = dataconfig.model_id.lower()
        if 'gemma' in model_id:
            return "gemmascope"
        elif 'llama' in model_id:
            return "llamascope"
        elif 'pythia' in model_id:
            return "ssae"

    return None


def main(args):
    """Evaluate cosine similarities using specified method."""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load test data
    loader = TestDataLoader(device=device, verbose=args.verbose)
    tensors, dataconfig, test_labels, status = loader.load_test_data(
        args.data, args.dataconfigpath
    )

    if tensors is None:
        print(f"Failed to load test data: {status}")
        return

    tilde_z_test, z_test = tensors
    print(f"Data loaded successfully: {status}")

    results = {}

    if args.method in ["decoder", "both"]:
        print("\n=== Decoder-based Cosine Similarities ===")

        # Split data by concept
        concept_test_sets = loader.split_by_label(
            tilde_z=tilde_z_test, z=z_test, labels=test_labels
        )

        if concept_test_sets is None:
            print("Failed to split data by concepts")
            return

        # Auto-detect model type if not specified
        modeltype = args.modeltype
        if not modeltype:
            detected_type = detect_model_type_from_config(dataconfig)
            if detected_type:
                modeltype = detected_type
                print(f"Auto-detected model type: {modeltype}")
            else:
                raise ValueError("Could not auto-detect model type. Please specify --modeltype")

        # Load model parameters
        if modeltype == "llamascope":
            decoder_weight, decoder_bias, _, _ = load_llamascope_checkpoint()
        elif modeltype == "gemmascope":
            decoder_weight, decoder_bias, _, _ = load_gemmascope_checkpoint()
        elif modeltype == "ssae":
            if not args.modeldir:
                raise ValueError("--modeldir is required for SSAE model type")
            decoder_weight_matrices, decoder_bias_vectors, _, _ = (
                load_ssae_models([args.modeldir])
            )
            decoder_weight = decoder_weight_matrices[0]
            decoder_bias = decoder_bias_vectors[0]
        else:
            raise ValueError(
                "Invalid model type. Choose 'llamascope', 'gemmascope', or 'ssae'"
            )

        concept_metrics = {}
        for concept, concept_test_set in concept_test_sets.items():
            concept_tilde_z, concept_z = concept_test_set

            mean_cos, std_cos, steering_vector = (
                get_max_cos_and_steering_vector_for_concept(
                    concept_z, concept_tilde_z, decoder_weight, decoder_bias
                )
            )

            print(
                f"Concept {concept}: Mean = {mean_cos:.4f}, Std = {std_cos:.4f}"
            )

            concept_metrics[concept] = {
                "mean_cos": mean_cos,
                "std_cos": std_cos,
                "steering_vector": steering_vector,
            }

            # Save steering vector if requested
            if args.store_steering and steering_vector is not None:
                steering_dir = f"{os.path.dirname(args.data)}/steering_vector_{modeltype}"
                os.makedirs(steering_dir, exist_ok=True)

                steering_path = os.path.join(
                    steering_dir,
                    f"steering_vector_concept_{concept}_{modeltype}.pt",
                )
                torch.save(steering_vector, steering_path)
                print(
                    f"Saved steering vector for concept {concept} to {steering_path}"
                )

        results["decoder_cosines"] = concept_metrics

    if args.method in ["pca", "both"]:
        print("\n=== PCA-based Cosine Similarities ===")

        cosines_pca = take_pca(z_test, tilde_z_test, device)
        mean_pca = np.mean(cosines_pca)
        std_pca = np.std(cosines_pca)

        print(
            f"PCA cosine similarities: Mean = {mean_pca:.4f}, Std = {std_pca:.4f}"
        )

        results["pca_cosines"] = {
            "cosines": cosines_pca,
            "mean": mean_pca,
            "std": std_pca,
        }

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate cosine similarities"
    )
    parser.add_argument("data", help="Path to test data file")
    parser.add_argument(
        "dataconfigpath", help="Path to data configuration YAML"
    )
    parser.add_argument(
        "--method",
        choices=["decoder", "pca", "both"],
        default="both",
        help="Evaluation method to use",
    )
    parser.add_argument(
        "--modeltype",
        choices=["llamascope", "gemmascope", "ssae"],
        help="Type of model (auto-detected if not specified)",
    )
    parser.add_argument(
        "--modeldir", help="Path to model directory (required for SSAE)"
    )
    parser.add_argument(
        "--store-steering", action="store_true", help="Store steering vectors"
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Enable verbose output"
    )

    args = parser.parse_args()

    # Validation
    if args.method in ["decoder", "both"]:
        if args.modeltype == "ssae" and not args.modeldir:
            parser.error("--modeldir is required when using SSAE model type")

    main(args)
