from base import CODEBOOK
import random
import os
import argparse


def sample_excluding_elements(list, exclude):
    filtered_list = [item for item in list if item not in exclude]

    if not filtered_list:
        raise ValueError("No elements left to sample after excluding.")

    return random.choice(filtered_list)


def generate_clusterable_with_color_change(key, eval_length=100):
    color_1 = CODEBOOK["color"][key]
    color_2 = CODEBOOK["color"][key + 1]
    data = []

    for _ in range(eval_length):
        init_attribute = tilde_attribute = random.choice(CODEBOOK["attribute"])
        init_object = tilde_object = random.choice(CODEBOOK["object"])
        x = f"{init_attribute} {color_1} {init_object}"
        tilde_x = f"{tilde_attribute} {color_2} {tilde_object}"
        data.append([x, tilde_x, 1, 2])
    return data


def write_to_txt_file(filename, list_of_lists):
    with open(filename, "w") as file:
        for single_list in list_of_lists:
            row = ", ".join(map(str, single_list)) + "\n"
            file.write(row)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("embedding_dir")
    args = parser.parse_args()
    key_1, key_2 = random.sample(range(0, len(CODEBOOK) - 1), 2)
    data_1 = generate_clusterable_with_color_change(key_1)
    filename_1 = os.path.join(
        args.embedding_dir,
        "eval_gradeschooler_" + str(key_1) + "_" + ".txt",
    )
    write_to_txt_file(filename_1, data_1)
    data_2 = generate_clusterable_with_color_change(key_2)
    filename_2 = os.path.join(
        args.embedding_dir,
        "eval_gradeschooler_" + str(key_2) + "_" + ".txt",
    )
    write_to_txt_file(filename_2, data_2)
