from base import CODEBOOK
import random


def sample_excluding_elements(list, exclude):
    filtered_list = [item for item in list if item not in exclude]

    if not filtered_list:
        raise ValueError("No elements left to sample after excluding.")

    return random.choice(filtered_list)


def generate_clusterable_with_color_change(eval_length=100):
    color_1 = CODEBOOK["color"][0]
    color_2 = CODEBOOK["color"][1]
    data = []

    for _ in range(eval_length):
        init_attribute = tilde_attribute = random.choice(CODEBOOK["attribute"])
        init_object = tilde_object = random.choice(CODEBOOK["object"])
        num_tfs = random.randint(0, len(CODEBOOK) - 1)
        factors_to_resample = random.sample(["attribute", "object"], num_tfs)
        tf_ids = []
        for factor in factors_to_resample:
            if factor == "attribute":
                exclude = init_attribute
                tilde_attribute = sample_excluding_elements(
                    CODEBOOK[factor], exclude=exclude
                )
                tf_ids.append(1)
            elif factor == "object":
                exclude = init_object
                tilde_object = sample_excluding_elements(
                    CODEBOOK[factor], exclude=exclude
                )
                tf_ids.append(3)
            else:
                raise ValueError
        x = f"{init_attribute} {color_1} {init_object}"
        tilde_x = f"{tilde_attribute} {color_2} {tilde_object}"
        data.append([x, tilde_x, num_tfs, tf_ids])
    return data
