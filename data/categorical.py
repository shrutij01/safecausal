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


def generate_counterfactual_pair():
    init_shape = tilde_shape = random.choice(CATEGORICAL_CODEBOOK["shape"])
    init_color = tilde_color = random.choice(CATEGORICAL_CODEBOOK["color"])
    init_object = tilde_object = random.choice(CATEGORICAL_CODEBOOK["object"])
    num_tfs = random.randint(1, len(CATEGORICAL_CODEBOOK))
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


def generate_data(size):
    dataset = []
    for _ in range(size):
        cf_pair = generate_counterfactual_pair()
        dataset.append(cf_pair)
    return dataset


def write_to_python_file(filename, list_of_lists):
    with open(filename, "w", buffering=2048) as file:
        file.write("data = [\n")
        for single_list in list_of_lists:
            tuple_str = f"({repr(single_list[0])}, {repr(single_list[1])}),\n"
            file.write(tuple_str)
        file.write("]\n")


if __name__ == "__main__":
    datadir = "/home/mila/j/joshi.shruti/causalrepl_space/psp/data/"
    filename = os.path.join(datadir, "categorical_base.py")
    size = 30000
    dataset = generate_data(size)
    write_to_python_file(filename, dataset)
