import psp.data_utils as utils

import argparse
import torch
import torch.nn.functional as F

import numpy as np
from safetensors.torch import load_file
from huggingface_hub import hf_hub_download
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def first_pca_direction(X: torch.Tensor) -> torch.Tensor:
    """
    Computes the first principal direction of X.

    Args:
        X (torch.Tensor): Input data of shape (n_samples, n_features)

    Returns:
        direction (torch.Tensor): First principal direction (unit vector of shape [n_features])
    """
    X_centered = X - X.mean(dim=0, keepdim=True)
    _, _, Vh = torch.linalg.svd(X_centered, full_matrices=False)
    first_direction = Vh[0]  # This is already normalized
    return utils.numpify(first_direction)


def load_llamascope_checkpoint():
    model_id = "fnlp/Llama3_1-8B-Base-LXR-8x"

    filename = "Llama3_1-8B-Base-L3R-8x/checkpoints/consolidated.safetensors"

    filepath = hf_hub_download(
        repo_id=model_id,
        filename=filename,
        local_dir="checkpoints",
        local_dir_use_symlinks=False,
    )

    print(f"Downloaded checkpoint to: {filepath}")

    # 2. Load the safetensor
    state_dict = load_file(filepath)
    decoder_weight = state_dict[
        "decoder.weight"
    ]  # shape: (vocab_size, d_model)
    print(f"decoder.weight shape: {decoder_weight.shape}")

    return decoder_weight


def max_cosine_similarity(
    z: torch.Tensor, z_tilde: torch.Tensor, decoder_weight: torch.Tensor
):
    """
    z: [B, D] tensor of original vectors
    decoder_weight: [V, D] decoder weight matrix (rows = token embeddings)

    Returns:
        mean and std of max cosine similarities over decoder directions
    """
    z = F.normalize(z, dim=1)  # [B, D]
    decoder = F.normalize(
        decoder_weight, dim=1
    ).T  # [D, V] — columns as directions

    B, D = z.shape
    V = decoder.shape[1]

    # z: [B, D], decoder: [D, V]
    z_tilde_hat = z.unsqueeze(2) + decoder.unsqueeze(0)  # [B, D, V]
    z_tilde_hat = F.normalize(
        z_tilde_hat, dim=1
    )  # normalize shifted vectors: [B, D, V]

    z_tilde = z_tilde.unsqueeze(2)  # [B, D, 1]
    cosines = torch.bmm(z_tilde.transpose(1, 2), z_tilde_hat).squeeze(
        1
    )  # [B, V]

    max_cosines = cosines.max(dim=1).values  # [B]
    return max_cosines.mean().item(), max_cosines.std().item()


def main(args):
    tilde_z_test, z_test = utils.load_test_data(
        data_file=args.data_file,
    )
    shifts = utils.tensorify((tilde_z_test - z_test), device)
    pca_vec = first_pca_direction(shifts)
    z_pca = z_test / np.linalg.norm(z_test) + pca_vec
    cosines_pca = []
    for i in range(tilde_z_test.shape[0]):
        cosines_pca.append(
            cosine_similarity(
                tilde_z_test[i].reshape(1, -1), z_pca[i].reshape(1, -1)
            )
        )
    import ipdb

    ipdb.set_trace()

    print("Computing max cosine similarities...")
    decoder_weight = load_llamascope_checkpoint()
    max_cosine_similarity(
        utils.tensorify(z_test, device),
        utils.tensorify(tilde_z_test, device),
        decoder_weight,
    )

    import ipdb

    ipdb.set_trace()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("data_file")
    parser.add_argument("dataconfig_file")
    args = parser.parse_args()
    main(args)
