import random
import copy

import os
import datetime

CODEBOOK = {
    "object": [
        "Sofa",
        "Lamp",
        "Microwave",
        "Television",
        "Chair",
        "Bookshelf",
        "Clock",
        "Towel",
        "Kettle",
    ],
    "color": [
        "Orange",
        "Purple",
        "Pink",
        "Magenta",
        "Teal",
        "Maroon",
        "Olive",
        "Navy",
        "Lavender",
        "Turquoise",
    ],
    "attribute": [
        "Vintage",
        "Loud",
        "Bohemian",
        "Contemporary",
        "Strong",
        "Weak",
        "Deep",
        "Shallow",
        "Ancient",
        "Modern",
    ],
}


def sample_excluding_elements(list, exclude):
    filtered_list = [item for item in list if item not in exclude]

    if not filtered_list:
        raise ValueError("No elements left to sample after excluding.")

    return random.choice(filtered_list)


def generate_counterfactual_pair():
    init_attribute = tilde_attribute = random.choice(CODEBOOK["attribute"])
    init_color = tilde_color = random.choice(CODEBOOK["color"])
    init_object = tilde_object = random.choice(CODEBOOK["object"])
    num_tfs = random.randint(1, len(CODEBOOK))
    tf_ids = []
    factors_to_resample = random.sample(list(CODEBOOK.keys()), num_tfs)
    for factor in factors_to_resample:
        if factor == "attribute":
            exclude = init_attribute
            tilde_attribute = sample_excluding_elements(
                CODEBOOK[factor], exclude=exclude
            )
            tf_ids.append(1)
        elif factor == "color":
            exclude = init_color
            tilde_color = sample_excluding_elements(
                CODEBOOK[factor], exclude=exclude
            )
            tf_ids.append(2)
        elif factor == "object":
            exclude = init_object
            tilde_object = sample_excluding_elements(
                CODEBOOK[factor], exclude=exclude
            )
            tf_ids.append(3)
        else:
            raise ValueError
    x = f"{init_attribute} {init_color} {init_object}"
    tilde_x = f"{tilde_attribute} {tilde_color} {tilde_object}"
    return [x, tilde_x, num_tfs, tf_ids]


def generate_data(size):
    dataset = []
    for _ in range(size):
        cf_pair = generate_counterfactual_pair()
        dataset.append(cf_pair)
    return dataset


def write_to_txt_file(filename, list_of_lists):
    with open(filename, "w") as file:
        for single_list in list_of_lists:
            row = ", ".join(map(str, single_list)) + "\n"
            file.write(row)


if __name__ == "__main__":
    datadir = "/network/scratch/j/joshi.shruti/psp/gradeschooler"
    current_datetime = datetime.datetime.now()
    timestamp_str = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")
    filename = os.path.join(datadir, timestamp_str, "gradeschooler.txt")
    size = 10000
    dataset = generate_data(size)
    write_to_txt_file(filename, dataset)
