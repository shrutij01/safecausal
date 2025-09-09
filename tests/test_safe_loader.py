#!/usr/bin/env python3
"""
Test script for SafeDataLoader

Usage:
    python test_safe_loader.py /path/to/datafile /path/to/dataconfig.yaml
"""

import sys
import torch
from loaders import SafeDataLoader


def test_safe_data_loading(datafile: str, dataconfig_path: str):
    """Test the SafeDataLoader with real data."""

    print("🧪 Testing SafeDataLoader")
    print("=" * 50)

    # Initialize loader
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader = SafeDataLoader(device=device, verbose=True)

    # Load data safely
    print("\n📥 Loading data with SafeDataLoader...")
    result = loader.load_and_validate_test_data(datafile, dataconfig_path)

    tensor_data, dataconfig, concept_labels, status = result

    if tensor_data is None:
        print(f"❌ Data loading failed: {status}")
        return False

    tilde_z_test, z_test = tensor_data

    print(f"\n✅ Data loading successful!")
    print(f"   Status: {status}")
    print(f"   z_test shape: {z_test.shape}, device: {z_test.device}")
    print(
        f"   tilde_z_test shape: {tilde_z_test.shape}, device: {tilde_z_test.device}"
    )
    print(f"   Concept labels: {len(concept_labels)} labels")
    print(f"   Data config rep_dim: {dataconfig.rep_dim}")

    # Test concept splitting
    print("\n🔄 Testing concept splitting...")
    concept_splits = loader.split_data_by_concept(
        tilde_z_test, z_test, concept_labels
    )

    if concept_splits is None:
        print("❌ Concept splitting failed")
        return False

    print(f"✅ Concept splitting successful!")
    print(f"   Number of concepts: {len(concept_splits)}")
    for concept, (tilde_subset, z_subset) in concept_splits.items():
        print(f"   Concept {concept}: {z_subset.shape[0]} samples")

    # Test safe normalization
    print("\n🔄 Testing safe normalization...")
    from loaders.validation import safe_normalize

    z_normalized, norm_status = safe_normalize(z_test, dim=1, name="z_test")
    if z_normalized is None:
        print(f"❌ Normalization failed: {norm_status}")
        return False

    print(f"✅ Normalization successful!")
    print(f"   Normalized tensor shape: {z_normalized.shape}")
    print(
        f"   Normalized tensor range: [{z_normalized.min():.6f}, {z_normalized.max():.6f}]"
    )

    return True


def main():
    if len(sys.argv) != 3:
        print("Usage: python test_safe_loader.py <datafile> <dataconfig.yaml>")
        print(
            "Example: python test_safe_loader.py /path/to/test.h5 /path/to/config.yaml"
        )
        sys.exit(1)

    datafile = sys.argv[1]
    dataconfig_path = sys.argv[2]

    try:
        success = test_safe_data_loading(datafile, dataconfig_path)
        if success:
            print(
                "\n🎉 All tests passed! SafeDataLoader is working correctly."
            )
        else:
            print("\n💥 Some tests failed. Check the error messages above.")
            sys.exit(1)

    except Exception as e:
        print(f"\n💥 Test script failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
