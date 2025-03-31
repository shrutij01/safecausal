import psp.data_utils as utils

import argparse
import torch
import torch.nn.functional as F

import numpy as np
from safetensors.torch import load_file
from huggingface_hub import hf_hub_download
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def svd_flip(u, v):
    # columns of u, rows of v
    max_abs_cols = torch.argmax(torch.abs(u), 0)
    i = torch.arange(u.shape[1]).to(u.device)
    signs = torch.sign(u[max_abs_cols, i])
    u *= signs
    v *= signs.view(-1, 1)
    return u, v


def pca_transform(X: torch.Tensor):
    """
    Computes the first principal direction of X.

    Args:
        X (torch.Tensor): Input data of shape (n_samples, n_features)

    Returns:
        direction (torch.Tensor): First principal direction (unit vector of shape [n_features])
    """
    n, d = X.shape
    mean = X.mean(dim=0, keepdim=True)
    X_centered = X - mean
    U, _, Vh = torch.linalg.svd(X_centered, full_matrices=False)
    U, Vh = svd_flip(U, Vh)
    components = Vh
    return torch.matmul(X - mean, components.t()), components, mean


def load_llamascope_checkpoint():
    model_id = "fnlp/Llama3_1-8B-Base-LXR-8x"

    filename = "Llama3_1-8B-Base-L31R-8x/checkpoints/final.safetensors"

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
    z_tilde = F.normalize(z_tilde, dim=1)  # [B, D]
    decoder = F.normalize(
        decoder_weight, dim=1
    )  # [V, D] — columns as directions
    decoder = decoder.to(z.device)

    B, D = z.shape
    V = decoder.shape[0]

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

    shifts_transformed, components, mean = pca_transform(shifts.float())
    pca_vec = (
        (components.sum(dim=0, keepdim=True) + mean).mean(0)
        # .view(z_test[0].shape[0], z_test[0].shape[1])
    )
    z_test = utils.tensorify(z_test, device)
    z_pca = F.normalize(z_test) + pca_vec
    z_pca = F.normalize(z_pca)
    z_pca = utils.numpify(z_pca)
    cosines_pca = []
    for i in range(tilde_z_test.shape[0]):
        cosines_pca.append(
            cosine_similarity(
                tilde_z_test[i].reshape(1, -1), z_pca[i].reshape(1, -1)
            )
        )
    decoder_weight = load_llamascope_checkpoint()
    mean_scope, std_scope = max_cosine_similarity(
        z_test,
        utils.tensorify(tilde_z_test, device),
        decoder_weight,
    )
    print(
        "USING PCA cosine similarities",
        np.mean(cosines_pca),
        np.std(cosines_pca),
    )

    print("AND using Llamascope, computing max cosine similarities...")
    print("...", mean_scope, std_scope)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("data_file")
    parser.add_argument("dataconfig_file")
    args = parser.parse_args()
    main(args)
