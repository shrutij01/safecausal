import random
from types import GenericAlias
from typing import Optional, List, Dict
import pandas as pd
import argparse
import datetime
import yaml
import os
import itertools
import numpy as np

import torch


def generate_invertible_matrix(size):
    while True:
        matrix = np.random.randint(1, size, (size, size))
        if np.linalg.det(matrix) != 0:
            return matrix


def generate_overlapping_block_binary_vectors(
    num_tuples, travellers_K, travellers_N
):

    def generate_binary_vectors(N, K):
        """Generates all binary vectors of length N with exactly K ones."""
        # Generate all combinations of indices to set to 1
        indices = list(itertools.combinations(range(N), K))
        # Initialize vectors
        vectors = []
        for index_set in indices:
            # Create a zero vector and set specific indices to 1
            vector = [0] * N
            for index in index_set:
                vector[index] = 1
            vectors.append(vector)
        return vectors

    def sample_and_sum_vectors(vectors, N):
        """Samples 1 to N vectors from the given list and sums them if more than one is sampled."""
        # Randomly choose how many vectors to sample, between 1 and N
        num_samples = random.randint(1, N)
        # Sample without replacement
        sampled_vectors = random.sample(vectors, num_samples)
        # If only one vector is sampled, return it
        if num_samples == 1:
            return sampled_vectors[0], sampled_vectors
        # Otherwise, return the sum of the sampled vectors
        x = np.sum(sampled_vectors, axis=0).tolist()
        return x, sampled_vectors

    column_names = ["Tx", "x", "delta_C"]
    data: List[Dict] = []
    lin_ent_tf = generate_invertible_matrix(travellers_N)
    # Generate vectors and perform sampling
    binary_vectors = generate_binary_vectors(travellers_N, travellers_K)
    for _ in range(num_tuples):
        result, delta_c = sample_and_sum_vectors(binary_vectors, travellers_N)
        row = {
            "Tx": lin_ent_tf @ result,
            "x": result,
            "delta_C": delta_c,
        }
        data.append(row)
    df = pd.DataFrame(data, columns=column_names)
    return df


def generate_distinct_block_binary_vectors(
    num_tuples, travellers_K, travellers_N
):
    column_names = ["Tx", "x", "delta_C"]
    data: List[Dict] = []
    lin_ent_tf = generate_invertible_matrix(travellers_N * travellers_K)
    for _ in range(num_tuples):
        total_length = travellers_N * travellers_K
        vector = np.zeros(total_length, dtype=int)

        # Randomly choose how many blocks to fill with ones (1 to K blocks)
        num_blocks_to_fill = np.random.randint(1, travellers_K + 1)

        # Randomly choose which blocks to fill
        blocks_to_fill = np.random.permutation(travellers_N)[
            :num_blocks_to_fill
        ]

        # Set the chosen blocks to all ones
        for block_index in blocks_to_fill:
            start_idx = block_index * travellers_N
            end_idx = start_idx + travellers_N
            vector[start_idx:end_idx] = 1
        row = {"Tx": lin_ent_tf @ vector, "x": vector, "delta_C": vector}
        data.append(row)
    df = pd.DataFrame(data, columns=column_names)
    return df


def generate_binary_vectors(travellers_K, num_tuples):
    # Initialize an array to hold all vectors
    vectors = np.zeros((num_tuples, travellers_K), dtype=int)

    # Generate random number of ones for each vector
    ones_counts = np.random.randint(1, travellers_K + 1, size=num_tuples)

    # Populate the vectors with ones
    for i in range(num_tuples):
        indices = np.random.choice(travellers_K, ones_counts[i], replace=False)
        vectors[i, indices] = 1

    # Generate a linear transformation matrix
    lin_ent_tf = generate_invertible_matrix(travellers_K)

    # Apply the transformation
    transformed_vectors = np.dot(vectors, lin_ent_tf.T)
    data = {
        "Tx": list(transformed_vectors),
        "x": list(vectors),
        "delta_C": list(vectors),
    }

    df = pd.DataFrame(data)
    return df


def main(args):
    if args.dgp == 1:
        data_df = generate_binary_vectors(args.num_tuples, args.travellers_K)
    elif args.dgp == 2:
        data_df = generate_distinct_block_binary_vectors(
            args.num_tuples, args.travellers_K, args.travellers_N
        )
    elif args.dgp == 3:
        data_df = generate_overlapping_block_binary_vectors(
            args.num_tuples, args.travellers_K, args.travellers_N
        )
    else:
        raise ValueError
    current_datetime = datetime.datetime.now()
    timestamp_str = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")
    dir_location = "/network/scratch/j/joshi.shruti/psp/travellers/"
    directory_name = os.path.join(dir_location, timestamp_str)
    if not os.path.exists(directory_name):
        os.makedirs(directory_name)
    df_location = os.path.join(
        directory_name, "travellers" + str(args.dgp) + ".csv"
    )
    if os.path.exists(df_location):
        overwrite = input(
            "A dataset already exists at {}. Do you want to overwrite it? (yes/no): ".format(
                df_location
            )
        )
        if overwrite.lower() != "yes":
            print("Skipping dataset creation and saving.")
            exit()

    if args.dgp == 1:
        dataset_name = "synth1"
    elif args.dgp == 2:
        dataset_name = "synth2"
    elif args.dgp == 3:
        dataset_name = "synth3"
    else:
        raise ValueError
    config = {
        "dataset_name": dataset_name,
        "size": args.num_tuples,
        "dgp": args.dgp,
        "travellers_K": args.travellers_K,
        "travellers_N": args.travellers_N,
        "cfc_column_names": ["Tx", "x", "delta_C"],
        "split": 0.9,
    }
    config_path = os.path.join(directory_name, "data_config.yaml")
    with open(config_path, "w") as file:
        yaml.dump(config, file)

    data_df.to_csv(df_location)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--num-tuples", type=int, default=100000)
    parser.add_argument("--travellers-K", type=int, default=2)
    parser.add_argument("--travellers-N", type=int, default=3)
    parser.add_argument("--dgp", type=int, default=1, choices=[1, 2, 3])

    args = parser.parse_args()
    main(args)
