import random
from typing import Optional, List, Dict
import pandas as pd
import argparse
import datetime
import yaml
import os


def translate_objects_single_coordinate(
    num_tuples: int,
    total_objects: int,
    max_grid_limit: int,
    num_objects_to_translate: Optional[int] = None,
    amount_to_translate: Optional[int] = None,
    max_amount_to_translate: int = 5,
) -> pd.DataFrame:
    assert (
        num_tuples >= total_objects
    ), f"need to generate more tuples {num_tuples} than number of objects {total_objects}"
    if num_objects_to_translate:
        assert (
            num_objects_to_translate <= total_objects
        ), f"if passing how many objects to translate {num_objects_to_translate}, can't exceed total_objects {total_objects}"

    object_position_columns = [
        f"object_{i}_position" for i in range(total_objects)
    ]
    object_translation_columns = [
        f"translation_for_object_{i}" for i in range(total_objects)
    ]
    fixed_columns = [
        "step",
        "num_objects_translated",
        "ids_objects_translated",
    ]
    all_column_names = (
        fixed_columns + object_position_columns + object_translation_columns
    )
    data: List[Dict] = []

    # Generate initial coordinates
    positions = [
        [random.randint(0, max_grid_limit), random.randint(0, max_grid_limit)]
        for _ in range(total_objects)
    ]
    translations_along_axes = [[0, 0] for _ in range(total_objects)]
    row = {
        "step": 0,
        "num_objects_translated": 0,
        "ids_objects_translated": None,
    }
    object_positions = {
        f"object_{idx}_position": positions[idx][:]
        for idx in range(total_objects)
    }
    translations = {
        f"translation_for_object_{idx}": translations_along_axes[idx][:]
        for idx in range(total_objects)
    }
    row.update(object_positions)
    row.update(translations)
    data.append(row)

    for step in range(0, num_tuples):
        # 1. first indeterminacy: num objects
        current_num_objects_to_translate = (
            num_objects_to_translate
            if num_objects_to_translate is not None
            else random.randint(1, total_objects)
        )
        # 2. which objects
        objects_to_translate = random.sample(
            range(total_objects), current_num_objects_to_translate
        )
        # 3. and by how much. this is fixed for now for every object.
        translations_along_axes = [[0, 0] for _ in range(total_objects)]
        for obj in objects_to_translate:
            axis = random.randint(0, 1)
            if amount_to_translate is None:
                translations_along_axes[obj][axis] += random.randint(
                    1, max_amount_to_translate
                )
            else:
                translations_along_axes[obj][axis] += amount_to_translate

        positions = [
            [a + b for a, b in zip(sublist1, sublist2)]
            for sublist1, sublist2 in zip(positions, translations_along_axes)
        ]

        row = {
            "step": step + 1,
            "num_objects_translated": current_num_objects_to_translate,
            "ids_objects_translated": objects_to_translate,
        }
        object_positions = {
            f"object_{idx}_position": positions[idx][:]
            for idx in range(total_objects)
        }
        translations = {
            f"translation_for_object_{idx}": translations_along_axes[idx][:]
            for idx in range(total_objects)
        }
        row.update(object_positions)
        row.update(translations)
        data.append(row)

    df = pd.DataFrame(data, columns=all_column_names)

    return df

    return positions[:num_tuples]


def main(args):

    positions_df = translate_objects_single_coordinate(
        num_tuples=args.num_tuples,
        total_objects=args.total_objects,
        max_grid_limit=args.max_grid_limit,
        num_objects_to_translate=args.num_objects_to_translate,
        amount_to_translate=args.amount_to_translate,
        max_amount_to_translate=args.max_amount_to_translate,
    )
    current_datetime = datetime.datetime.now()
    timestamp_str = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")
    dir_location = "/network/scratch/j/joshi.shruti/psp/toy_translator/"
    directory_name = os.path.join(dir_location, timestamp_str)
    if not os.path.exists(directory_name):
        os.makedirs(directory_name)
    df_location = os.path.join(
        directory_name, "multi_objects_single_coordinate.csv"
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
    object_translation_columns = [
        f"translation_for_object_{i}" for i in range(args.total_objects)
    ]
    config = {
        "num_objects": args.total_objects,
        "cfc_column_names": object_translation_columns,
        "split": 0.9,
        "size": args.total_objects,
    }
    config_path = os.path.join(directory_name, "config.yaml")
    with open(config_path, "w") as file:
        yaml.dump(config, file)

    positions_df.to_csv(df_location)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--num-tuples", type=int, default=1000)
    parser.add_argument("--total-objects", type=int, default=3)

    parser.add_argument("--max-grid-limit", default=10)
    parser.add_argument("--num-objects-to-translate", default=None)
    parser.add_argument("--amount-to-translate", default=None)
    parser.add_argument("--max-amount-to-translate", default=1)

    args = parser.parse_args()
    main(args)
