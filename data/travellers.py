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
import math
import pickle


def generate_invertible_matrix(size):
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
    return df, np.array(binary_vectors)


def generate_distinct_block_binary_vectors(
    num_tuples, travellers_K, travellers_N
):
    column_names = ["Tx", "x", "delta_C"]
    data: List[Dict] = []
    total_length = travellers_N * travellers_K
    lin_ent_tf = generate_invertible_matrix(travellers_N * travellers_K)
    for _ in range(num_tuples):
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
    global_C = np.zeros((total_length, travellers_K))
    for col in range(travellers_K):
        start_index = travellers_N * col
        end_index = travellers_N * (col + 1)
        global_C[start_index:end_index, col] = 1
    return df, global_C


def generate_binary_vectors(travellers_K, num_tuples):
    column_names = ["Tx", "x", "delta_C"]
    data = []
    lin_ent_tf = generate_invertible_matrix(travellers_K)

    # Initialize an array to hold all vectors
    vectors = np.zeros((num_tuples, travellers_K), dtype=int)

    # Generate random number of ones for each vector
    ones_counts = np.random.randint(1, travellers_K + 1, size=num_tuples)
    # Populate the vectors with ones
    for i in range(num_tuples):
        indices = np.random.choice(travellers_K, ones_counts[i], replace=False)
        vectors[i, indices] = 1
        transformed_vector = np.dot(vectors[i], lin_ent_tf.T)
        row = {
            "Tx": transformed_vector,
            "x": vectors[i],
            "delta_C": vectors[i],
        }
        data.append(row)

    df = pd.DataFrame(data, columns=column_names)
    global_C = np.eye(travellers_K)
    return df, global_C


def main(args):
    if args.dgp == 1:
        data_df, global_C = generate_binary_vectors(
            args.travellers_K, args.num_tuples
        )
        rep_dim = args.travellers_K
        num_concepts = args.travellers_K
    elif args.dgp == 2:
        data_df, global_C = generate_distinct_block_binary_vectors(
            args.num_tuples, args.travellers_K, args.travellers_N
        )
        rep_dim = args.travellers_K * args.travellers_N
        num_concepts = args.travellers_K
    elif args.dgp == 3:
        data_df, global_C = generate_overlapping_block_binary_vectors(
            args.num_tuples, args.travellers_K, args.travellers_N
        )
        rep_dim = args.travellers_N
        num_concepts = math.comb(args.travellers_N, args.travellers_K)
    else:
        raise ValueError

    if args.dgp == 1:
        dataset_name = "synth1"
    elif args.dgp == 2:
        dataset_name = "synth2"
    elif args.dgp == 3:
        dataset_name = "synth3"
    else:
        raise ValueError
    current_datetime = datetime.datetime.now()
    timestamp_str = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")
    dir_location = "/network/scratch/j/joshi.shruti/psp/travellers/"
    directory_name = os.path.join(dir_location, timestamp_str)
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
    pickle_path = os.path.join(directory_name, "global_C.pkl")
    with open(pickle_path, "wb") as file:
        pickle.dump(global_C, file)
    data_df.to_csv(df_location)
    config = {
        "dataset_name": dataset_name,
        "size": args.num_tuples,
        "dgp": args.dgp,
        "rep_dim": rep_dim,
        "num_concepts": num_concepts,
        "travellers_K": args.travellers_K,
        "travellers_N": args.travellers_N,
        "cfc_column_names": ["Tx", "x", "delta_C"],
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
    parser.add_argument("--travellers-K", type=int, default=2)
    parser.add_argument("--travellers-N", type=int, default=3)
    parser.add_argument("--dgp", type=int, default=1, choices=[1, 2, 3])

    args = parser.parse_args()
    main(args)
