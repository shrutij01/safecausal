import psp.data_utils as utils

import argparse
import torch
import numpy as np
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
    return first_direction


def main(args):
    tilde_z, z = utils.load_test_data(
        data_file=args.data_file,
    )
    shifts = utils.tensorify((tilde_z - z), device)
    pca_vec = first_pca_direction(shifts)
    z_pca = z / np.linalg.norm(z) + pca_vec
    cosines_pca = []
    for i in range(tilde_z.shape[0]):
        cosines_pca.append(
            cosine_similarity(
                tilde_z[i].reshape(1, -1), z_pca[i].reshape(1, -1)
            )
        )
    import ipdb

    ipdb.set_trace()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("data_file")
    parser.add_argument("dataconfig_file")
