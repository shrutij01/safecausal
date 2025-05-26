import numpy as np
import pandas as pd
from typing import Tuple, Optional

import argparse
import datetime
import os
import pickle
import yaml


def generate_invertible_matrix(size: int) -> np.ndarray:
    """
    Generates an invertible matrix (non-singular matrix) of a given size.
    For size 2, it returns a rotation matrix. For other sizes, it returns
    a randomly generated matrix with non-zero determinant.

    Parameters:
    - size (int): The dimension of the square matrix to generate.

    Returns:
    - np.ndarray: An invertible matrix of dimensions (size, size).
    """
    if size == 2:
        theta = np.radians(30)
        c, s = np.cos(theta), np.sin(theta)
        matrix = np.array(((c, -s), (s, c)))
        return matrix
    else:
        while True:
            matrix = np.random.randint(1, size, (size, size))
            if np.linalg.det(matrix) != 0:
                return matrix


def generate_binary_vectors(
    num_vectors: int, vector_size: int, max_k_ones: int
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Generates a specified number of binary vectors of a given length with exactly k ones,
    possibly including repetitions if the requested number exceeds the number of unique combinations.

    Parameters:
    - num_vectors (int): The number of vectors to generate.
    - vector_size (int): The size of each vector.
    - k_ones (int): The exact number of ones in each vector.

    Returns:
    - Tuple[pd.DataFrame, np.ndarray]: A tuple containing a DataFrame with deltaz and deltac
    vectors, and the ground truth deltac matrix.
    """
    vectors = []
    # Sample vectors with k-out-of-n ones
    for _ in range(num_vectors):
        vector = np.zeros(vector_size, dtype=int)
        # Randomly decide the number of ones, which can be anywhere from 0 to k
        ones_count = np.random.randint(0, max_k_ones + 1)
        indices = np.random.choice(vector_size, ones_count, replace=False)
        vector[indices] = 1
        vectors.append(vector)

    lin_ent_tf = generate_invertible_matrix(vector_size)

    # Apply the transformation and prepare the DataFrame
    transformed_vectors = np.dot(np.asarray(vectors), lin_ent_tf) 
    data = {
        "deltaz": transformed_vectors.tolist(), # transformed dense real-valued vectors (observations)
        "deltac": vectors, # sparse binary vectors (ground truth)
    }
    df = pd.DataFrame(data)

    # The ground truth deltac matrix
    gt_deltac = np.eye(vector_size) # use to measure how close learned transformation (learned_matrix * deltaz ≅ deltac) is to identity matrix 
    return df, gt_deltac


def generate_continuous_vectors(
    num_vectors: int,
    max_k: int,
    vector_size: int,
    distribution: str = "uniform",
    dist_params: Optional[dict] = None,
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Generates a specified number of continuous random vectors of a given length,
    where each vector is initially zero and has up to `max_k` randomly selected
    indices set to values sampled from a specified distribution (uniform or normal),
    and applies an invertible linear transformation to each vector.

    Parameters:
    - num_vectors (int): The number of vectors to generate.
    - vector_size (int): The size of each vector.
    - distribution (str): The type of distribution to use ('uniform' or 'normal').
    - dist_params (dict, optional): Parameters for the distribution; e.g.,
        {'low': 0, 'high': 1} for uniform, {'mean': 0, 'std': 1} for normal.

    Returns:
    - Tuple[pd.DataFrame, np.ndarray]: A tuple containing a DataFrame with deltaz and deltac
    vectors, and the ground truth deltac matrix.
    """
    vectors = np.zeros((num_vectors, vector_size), dtype=float)
    if distribution == "uniform":
        # Default parameters for uniform distribution
        low = dist_params.get("low", 0) if dist_params else 0
        high = dist_params.get("high", 1) if dist_params else 1
        for i in range(num_vectors):
            concepts_varied = np.random.randint(0, max_k + 1)
            indices = np.random.choice(
                vector_size, concepts_varied, replace=False
            )
            vectors[i, indices] = np.random.uniform(
                low, high, size=concepts_varied
            )
    elif distribution == "normal":
        # Default parameters for normal distribution
        mean = dist_params.get("mean", 0) if dist_params else 0
        std = dist_params.get("std", 1) if dist_params else 1
        for i in range(num_vectors):
            concepts_varied = np.random.randint(0, max_k + 1)
            indices = np.random.choice(
                vector_size, concepts_varied, replace=False
            )
            vectors[i, indices] = np.random.normal(
                loc=mean, scale=std, size=concepts_varied
            )
    else:
        raise ValueError(
            "Unsupported distribution type. Use 'uniform' or 'normal'."
        )

    # Generate an invertible transformation matrix
    lin_ent_tf = generate_invertible_matrix(vector_size)

    # Apply the transformation and prepare the DataFrame
    transformed_vectors = np.dot(vectors, lin_ent_tf)
    data = {
        "deltaz": transformed_vectors.tolist(),
        "deltac": vectors.tolist(),
    }
    df = pd.DataFrame(data)

    # The ground truth deltac matrix
    gt_deltac = np.eye(vector_size)
    return df, gt_deltac


def main(args):
    if args.dgp == 1:
        data_df, gt_deltac = generate_binary_vectors(
            num_vectors=args.num_tuples, vector_size=args.n, max_k_ones=args.k
        )
        dataset_name = "binsynth"
    elif args.dgp == 2:
        data_df, gt_deltac = generate_continuous_vectors(
            num_vectors=args.num_tuples,
            max_k=args.k,
            vector_size=args.n,
            distribution="uniform",
            dist_params={"low": 0, "high": 1},
        )
        dataset_name = "uniformsynth"
    elif args.dgp == 3:
        data_df, gt_deltac = generate_continuous_vectors(
            num_vectors=args.num_tuples,
            max_k=args.k,
            vector_size=args.n,
            distribution="normal",
            dist_params={"mean": 0, "std": 1},
        )
        dataset_name = "normalsynth"
    else:
        raise ValueError
    rep_dim = args.n
    num_concepts = args.n

    current_datetime = datetime.datetime.now()
    timestamp_str = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")
    dir_location = "/network/scratch/j/joshi.shruti/ssae/synthdata"
    directory_name = os.path.join(
        dir_location, str(args.dgp) + "_" + timestamp_str
    )
    if not os.path.exists(directory_name):
        os.makedirs(directory_name)
    df_location = os.path.join(directory_name, str(dataset_name) + ".csv")
    if os.path.exists(df_location):
        overwrite = input(
            "A dataset already exists at {}. Do you want to overwrite it? (yes/no): ".format(
                df_location
            )
        )
        if overwrite.lower() != "yes":
            print("Skipping dataset creation and saving.")
            exit()
    pickle_path = os.path.join(directory_name, "gt_deltac.pkl")
    with open(pickle_path, "wb") as file:
        pickle.dump(gt_deltac, file)
    data_df.to_csv(df_location)
    config = {
        "dataset": dataset_name,
        "size": args.num_tuples,
        "rep_dim": rep_dim,
        "num_concepts": num_concepts,
        "n": args.n,
        "k": args.k,
        "delta_z_column": "deltaz",
        "delta_c_column": "deltac",
        "train_split": 0.8,
        "eval_split": 0.9,
        "pickle_path": pickle_path,
    }
    config_path = os.path.join(directory_name, "data_config.yaml")
    with open(config_path, "w") as file:
        yaml.dump(config, file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--num-tuples", type=int, default=100000)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--n", type=int, default=9)
    parser.add_argument("--dgp", type=int, default=2, choices=[1, 2, 3])

    args = parser.parse_args()
    main(args)
