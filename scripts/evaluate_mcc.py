"""
Evaluate Mean Correlation Coefficients (MCCs) between SSAE model pairs.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import itertools
import numpy as np
import utils.metrics as metrics
from loaders import TestDataLoader, load_ssae_models


def compute_all_pairwise_mccs(weight_matrices: list) -> list[float]:
    """
    Compute mean correlation coefficients (MCCs) between all model pairs.
    """
    mccs = []
    for i, j in itertools.combinations(range(len(weight_matrices)), 2):
        mcc = metrics.mean_corr_coef(
            weight_matrices[i],
            weight_matrices[j],
            method="pearson",
        )
        mccs.append(mcc)
    return mccs


def main(args):
    """Evaluate MCCs between SSAE models."""
    # Load data
    loader = TestDataLoader(verbose=args.verbose)
    tensors, _, _, status = loader.load_test_data(
        args.datafile, args.dataconfigpath
    )

    if tensors is None:
        print(f"Failed to load test data: {status}")
        return

    print(f"Data loaded successfully: {status}")

    # Load SSAE models
    modeldirs = args.modeldirs
    if len(modeldirs) < 2:
        raise ValueError("You must provide at least two model directories.")

    (
        decoder_weight_matrices,
        _,
        _,
        _,
    ) = load_ssae_models(modeldirs)

    # Compute pairwise MCCs
    print("Computing pairwise MCCs...")
    mccs = compute_all_pairwise_mccs(decoder_weight_matrices)

    mean_mcc = np.mean(mccs)
    std_mcc = np.std(mccs)

    print("\nPairwise MCCs:")
    for i, (a, b) in enumerate(
        itertools.combinations(range(len(modeldirs)), 2)
    ):
        print(f"Model {a+1} vs Model {b+1}: MCC = {mccs[i]:.4f}")

    print(f"\nMean MCC: {mean_mcc:.4f}")
    print(f"Std  MCC: {std_mcc:.4f}")

    return {"mccs": mccs, "mean_mcc": mean_mcc, "std_mcc": std_mcc}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate MCCs between SSAE models"
    )
    parser.add_argument(
        "--datafile", required=True, help="Path to test data file"
    )
    parser.add_argument(
        "--dataconfigpath",
        required=True,
        help="Path to data configuration YAML",
    )
    parser.add_argument(
        "--modeldirs",
        nargs="+",
        type=str,
        required=True,
        help="List of model directories to compare (minimum 2)",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Enable verbose output"
    )

    args = parser.parse_args()
    main(args)
