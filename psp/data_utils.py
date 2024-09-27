import torch

from datasets import load_dataset as hf_load_dataset

from data import base, ana

import numpy as np
import os
import ast
import re
import pandas as pd
import h5py


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def tensorify(np_array, device):
    return torch.from_numpy(np_array).to(device).type(torch.float32)


def numpify(tensor):
    return tensor.detach().cpu().numpy()


def load_dataset(dataset_name, **kwargs):
    # if dataset_name == "ana":
    #     data = ana.generate_data(**kwargs)
    #     instruction = "Swap the letters in this string."
    #     cfc_tuples = data
    # elif dataset_name == "gradeschooler":
    #     cfc_tuples = []
    #     file_path = kwargs.get("file_path")
    #     if file_path is None:
    #         raise ValueError(
    #             "file_path must be provided as a keyword argument"
    #         )
    #     with open(file_path, "r") as f:
    #         context_pairs = [
    #             line.strip().split("\t") for line in f if line.strip()
    #         ]
    #     for cp in context_pairs:
    #         cfc_tuples.append(
    #             (cp[0].split(",")[0], cp[0].split(",")[1].lstrip())
    #         )
    #     instruction = "Change any one, or two, or all three of the entities in the string, and mention how many and which you changed."
    if dataset_name == "truthful_qa":
        import ipdb

        ipdb.set_trace()
        hf_dataset_name = "truthfulqa/truthful_qa"
        dataset_params = {
            "split": "validation",
            "name": "multiple_choice",
        }
        data = hf_load_dataset(hf_dataset_name, **dataset_params)
        # mc1_targets have a single correct choice and mc2_targets have
        # multiple choices that can be correct
        instruction = "Label as 0 for False and 1 for True."
        cfc_tuples = (
            [
                value
                for is_true, value in zip(
                    [d == 1 for d in data[0]["mc1_targets"]["labels"]],
                    data[0]["mc1_targets"]["choices"],
                )
                if is_true
            ],
            [
                value
                for is_true, value in zip(
                    [d == 0 for d in data[0]["mc1_targets"]["labels"]],
                    data[0]["mc1_targets"]["choices"],
                )
                if is_true
            ],
        )
    elif dataset_name == "binary_1":
        cfc_tuples = base.binary_1
        instruction = None
    else:
        raise NotImplementedError
    return cfc_tuples, instruction


def append_instruction(contexts, instruction):
    instruction_plus_contexts = []
    for context in contexts:
        instruction_plus_contexts.append(
            [str(instruction) + " " + str(context)]
        )
    return instruction_plus_contexts


def load_test_data(data_file):
    with h5py.File(data_file, "r") as f:
        cfc_embeddings_test = np.array(f["cfc_test"])
        tilde_z = cfc_embeddings_test[:, 1]
        z = cfc_embeddings_test[:, 0]
    return tilde_z, z


def get_md_steering_vector(data_file):
    with h5py.File(data_file, "r") as f:
        cfc_embeddings_train = np.array(f["cfc_train"])
        return np.mean(
            cfc_embeddings_train[:, 1] - cfc_embeddings_train[:, 0], axis=0
        )


def get_rep_pairs(num_pairs, data1, data2):
    assert data1.shape[0] == data2.shape[0]
    assert data1.shape[1] == data2.shape[1]
    total_samples = data1.shape[0]
    m = data1.shape[1]
    reps1 = np.zeros((num_pairs, m))
    reps2 = np.zeros((num_pairs, m))
    for i in range(num_pairs):
        id1, id2 = np.random.choice(total_samples, size=2, replace=False)
        reps1[i] = data1[id1]
        reps2[i] = data2[id2]
    return reps1, reps2


def archived_load_test_data(args, data_config):
    # todo: after test dict features
    delta_z = None
    tf_ids = []
    num_tfs = []
    if data_config.dataset == "toy_translator":
        data_type = data_config.dgp
        df_file = os.path.join(
            args.embedding_dir,
            "object_translations" + str(data_type) + ".csv",
        )
        cfc_columns = data_config.cfc_column_names
        converters = {col: ast.literal_eval for col in cfc_columns}
        x_df = pd.read_csv(df_file, converters=converters)
        x_df_test = x_df.iloc[int(data_config.split * data_config.size) :]

        def convert_to_list_of_ints(value):
            if isinstance(value, str):
                value = ast.literal_eval(value)
            return [int(delta_z) for delta_z in value]

        for column in x_df_test[cfc_columns]:
            x_df_test[column] = x_df_test[column].apply(
                convert_to_list_of_ints
            )
        # x_df_test["num_coordinates_translated"] = x_df_test[
        #     "num_coordinates_translated"
        # ].apply(convert_to_list_of_ints)
        # x_df_test["ids_coordinates_translated"] = x_df_test[
        #     "ids_coordinates_translated"
        # ].apply(convert_to_list_of_ints)

        delta_z = np.asarray(
            (
                x_df_test[cfc_columns]
                .apply(lambda row: sum(row, []), axis=1)
                .tolist()
            )
        )
        # num_tfs = np.asarray(x_df_test["num_coordinates_translated"].tolist())
        # tf_ids = np.asarray(x_df_test["ids_coordinates_translated"].tolist())
        return (
            delta_z,
            tf_ids,
            num_tfs,
        )

    elif data_config.dataset == "gradeschooler":
        embeddings_file = os.path.join(args.embedding_dir, "embeddings.h5")
        with h5py.File(embeddings_file, "r") as f:
            cfc1_embeddings_test = np.array(f["cfc1_test"])
            cfc2_embeddings_test = np.array(f["cfc2_test"])
        delta_z = cfc2_embeddings_test - cfc1_embeddings_test
        data_file = os.path.join(args.embedding_dir, "gradeschooler.txt")
        tf_ids = []
        num_tfs = []
        with open(data_file, "r") as f:
            context_pairs = [
                line.strip().split("\t") for line in f if line.strip()
            ]
        split = int(0.9 * data_config.dataset_length)
        for cp in context_pairs[split:]:
            match = re.search(r"\[(.*?)\]$", cp[0])
            if match:
                list_str = match.group(0)
                parsed_list = ast.literal_eval(list_str)
                int_list = [int(str(item).strip()) for item in parsed_list]
            else:
                raise ValueError
            int_list.sort()
            tf_id = int("".join(str(digit) for digit in int_list))
            tf_ids.append(tf_id)
            num_tfs.append(cp[0].split(",")[2])
            num_tfs = list(map(int, num_tfs))
            return (
                delta_z,
                tf_ids,
                num_tfs,
                cfc1_embeddings_test,
                cfc2_embeddings_test,
            )
    else:
        raise NotImplementedError(
            "Datasets implemented: toy_translator and gradeschooler"
        )


def load_eval_by_one_contrasts(args):
    eval_file_name_1 = "eval_embeddings_" + str(args.key_1) + ".h5"
    embeddings_file_1 = os.path.join(args.embedding_dir, eval_file_name_1)
    eval_file_name_2 = "eval_embeddings_" + str(args.key_2) + ".h5"
    embeddings_file_2 = os.path.join(args.embedding_dir, eval_file_name_2)
    with h5py.File(embeddings_file_1, "r") as f:
        cfc1_eval_1 = np.array(f["cfc1_eval"])
        cfc2_eval_1 = np.array(f["cfc2_eval"])
    with h5py.File(embeddings_file_2, "r") as f:
        cfc1_eval_2 = np.array(f["cfc1_eval"])
        cfc2_eval_2 = np.array(f["cfc2_eval"])
    return (cfc1_eval_1, cfc2_eval_1, cfc1_eval_2, cfc2_eval_2)


def get_embeddings_for_num_tfs(
    target_num_tfs, all_embeddings, all_num_tfs, all_tf_ids
):
    all_num_tfs_array = np.array(all_num_tfs)
    mask_num_tfs = all_num_tfs_array == target_num_tfs
    label_names = np.array(all_tf_ids)[mask_num_tfs]
    return all_embeddings[mask_num_tfs], label_names
