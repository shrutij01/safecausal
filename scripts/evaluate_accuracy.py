"""
Evaluate concept detection accuracy using encoder models.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import torch
import torch.nn.functional as F
from loaders import TestDataLoader, load_llamascope_checkpoint, load_ssae_models


def get_concept_detection_logits(z, encoder_weight, encoder_bias, decoder_bias, modeltype):
    """
    Compute the logits for concept detection.
    
    Args:
        z: [B, D] tensor of input vectors
        encoder_weight: Encoder weight matrix
        encoder_bias: Encoder bias vector
        decoder_bias: Decoder bias vector (unused but kept for compatibility)
        modeltype: Type of model ("llamascope" or "ssae")
    
    Returns:
        tuple: (mean_over_max_logits, entropy)
    """
    encoder_weight = encoder_weight.to(z.device)
    encoder_bias = encoder_bias.to(z.device)
    decoder_bias = decoder_bias.to(z.device)
    
    if modeltype == "llamascope":
        concept_projections = torch.nn.functional.relu(
            encoder_weight.to(torch.float32) @ z.T + encoder_bias.unsqueeze(1)
        ).T
    elif modeltype == "ssae":
        concept_projections = (
            encoder_weight.to(torch.float32) @ z.T + encoder_bias.unsqueeze(1)
        ).T
        # Uncomment if needed: concept_projections = (concept_projections > 0.1).float()
    else:
        raise ValueError("Invalid model type. Choose 'llamascope' or 'ssae'")
    
    eps = 1e-10
    concept_logits = torch.nn.functional.softmax(concept_projections, dim=1)
    
    # Element-wise multiplication of probabilities with their log
    entropy = -torch.sum(concept_logits * torch.log(concept_logits + eps), dim=1)
    mean_over_max_logits = concept_logits.max(dim=1).values.mean(dim=0)
    
    print(f"mean_over_max_logits: {mean_over_max_logits}")
    print(f"entropy - max: {entropy.max()}, min: {entropy.min()}, mean: {entropy.mean()}")
    
    return (mean_over_max_logits, entropy)


def main(args):
    """Evaluate concept detection accuracy."""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Load test data
    loader = TestDataLoader(device=device, verbose=args.verbose)
    tensors, dataconfig, test_labels, status = loader.load_test_data(
        args.datafile, args.dataconfig
    )
    
    if tensors is None:
        print(f"Failed to load test data: {status}")
        return
        
    tilde_z_test, z_test = tensors
    print(f"Data loaded successfully: {status}")
    
    # Load model parameters
    if args.modeltype == "llamascope":
        _, decoder_bias, encoder_weight, encoder_bias = load_llamascope_checkpoint()
        encoder_weight = encoder_weight.to(device)
        encoder_bias = encoder_bias.to(device)
        decoder_bias = decoder_bias.to(device)
        
    elif args.modeltype == "ssae":
        if not args.modeldir:
            raise ValueError("--modeldir is required for SSAE model type")
            
        _, decoder_bias_vectors, encoder_weight_matrices, encoder_bias_vectors = load_ssae_models([args.modeldir])
        encoder_weight = encoder_weight_matrices[0].to(device)
        encoder_bias = encoder_bias_vectors[0].to(device)
        decoder_bias = decoder_bias_vectors[0].to(device)
    else:
        raise ValueError("Invalid model type. Choose 'llamascope' or 'ssae'")
    
    # Compute concept detection accuracy
    mean_over_max_logits, entropy = get_concept_detection_logits(
        z_test, encoder_weight, encoder_bias, decoder_bias, args.modeltype
    )
    
    print(f"\n=== Concept Detection Results ===")
    print(f"Model type: {args.modeltype}")
    print(f"Mean over max logits: {mean_over_max_logits.item():.4f}")
    print(f"Entropy statistics:")
    print(f"  Max: {entropy.max().item():.4f}")
    print(f"  Min: {entropy.min().item():.4f}")
    print(f"  Mean: {entropy.mean().item():.4f}")
    print(f"  Std: {entropy.std().item():.4f}")
    
    results = {
        "mean_over_max_logits": mean_over_max_logits.item(),
        "entropy_max": entropy.max().item(),
        "entropy_min": entropy.min().item(),
        "entropy_mean": entropy.mean().item(),
        "entropy_std": entropy.std().item(),
    }
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate concept detection accuracy")
    parser.add_argument("--datafile", required=True, help="Path to test data file")
    parser.add_argument("--dataconfig", required=True, help="Path to data configuration YAML")
    parser.add_argument(
        "--modeltype", 
        required=True, 
        choices=["llamascope", "ssae"],
        help="Type of model used"
    )
    parser.add_argument("--modeldir", help="Path to model directory (required for SSAE)")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    
    args = parser.parse_args()
    
    if args.modeltype == "ssae" and not args.modeldir:
        parser.error("--modeldir is required when using SSAE model type")
    
    main(args)