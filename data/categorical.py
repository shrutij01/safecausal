import random
import copy

import os
import datetime

from base import CATEGORICAL_CODEBOOK


def sample_excluding_elements(list, exclude):
    filtered_list = [item for item in list if item not in exclude]

    if not filtered_list:
        raise ValueError("No elements left to sample after excluding.")

    return random.choice(filtered_list)


def generate_counterfactual_pair(dataset_type="train"):
    init_shape = tilde_shape = random.choice(CATEGORICAL_CODEBOOK["shape"])
    init_color = tilde_color = random.choice(CATEGORICAL_CODEBOOK["color"])
    init_object = tilde_object = random.choice(CATEGORICAL_CODEBOOK["object"])
    if dataset_type == "train":
        num_tfs = 1
    elif dataset_type == "test":
        num_tfs = random.randint(1, len(CATEGORICAL_CODEBOOK))
    else:
        raise ValueError
    tf_ids = []
    factors_to_resample = random.sample(
        list(CATEGORICAL_CODEBOOK.keys()), num_tfs
    )
    for factor in factors_to_resample:
        if factor == "shape":
            exclude = init_shape
            tilde_shape = sample_excluding_elements(
                CATEGORICAL_CODEBOOK[factor], exclude=exclude
            )
            tf_ids.append(1)
        elif factor == "color":
            exclude = init_color
            tilde_color = sample_excluding_elements(
                CATEGORICAL_CODEBOOK[factor], exclude=exclude
            )
            tf_ids.append(2)
        elif factor == "object":
            exclude = init_object
            tilde_object = sample_excluding_elements(
                CATEGORICAL_CODEBOOK[factor], exclude=exclude
            )
            tf_ids.append(3)
        else:
            raise ValueError
    x = f"{init_shape} {init_color} {init_object}"
    tilde_x = f"{tilde_shape} {tilde_color} {tilde_object}"
    return [x, tilde_x]


def generate_data(size, split=0.9):
    def generate_counterfactual_pairs(size, dataset_type):
        tuples = []
        for _ in range(size):
            cf_pair = generate_counterfactual_pair(dataset_type)
            tuples.append(cf_pair)
        return tuples

    cfc_train_tuples = generate_counterfactual_pairs(
        int(size * split), "train"
    )
    cfc_test_tuples = generate_counterfactual_pairs(
        int(size * (1 - split)), "test"
    )
    return cfc_train_tuples, cfc_test_tuples


def write_to_python_file(filename, train_data, test_data):
    with open(filename, "w", buffering=2048) as file:
        file.write("cfc_train_tuples = [\n")
        for pair in train_data:
            file.write(f"    ({repr(pair[0])}, {repr(pair[1])}),\n")
        file.write("]\n\n")

        file.write("cfc_test_tuples = [\n")
        for pair in test_data:
            file.write(f"    ({repr(pair[0])}, {repr(pair[1])}),\n")
        file.write("]\n")


if __name__ == "__main__":
    datadir = "/home/mila/j/joshi.shruti/causalrepl_space/ssae/data/"
    filename = os.path.join(datadir, "categorical_base.py")
    size = 30000
    cfc_train_tuples, cfc_test_tuples = generate_data(size)
    write_to_python_file(filename, cfc_train_tuples, cfc_test_tuples)
