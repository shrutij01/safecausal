import torch
from psp.sparse_dict import SparseDict

import argparse
import os
import h5py
import yaml
from box import Box
import numpy as np
import pandas as pd
import ast


def load_test_data(args, data_config):
    df_file = os.path.join(
        args.embedding_dir, "multi_objects_single_coordinate.csv"
    )
    cfc_columns = data_config.cfc_column_names
    converters = {col: ast.literal_eval for col in cfc_columns}
    x_df = pd.read_csv(df_file, converters=converters)
    x_df_test = x_df.iloc[int(data_config.split * data_config.size) :]

    def convert_to_list_of_ints(value):
        if isinstance(value, str):
            value = ast.literal_eval(value)
        return [int(x) for x in value]

    for column in x_df_test[cfc_columns]:
        x_df_test[column] = x_df_test[column].apply(convert_to_list_of_ints)

    x = np.asarray(
        (
            x_df_test[cfc_columns]
            .apply(lambda row: sum(row, []), axis=1)
            .tolist()
        )
    )
    return x


def main(args, device):
    data_config_file = os.path.join(args.embedding_dir, "config.yaml")
    with open(data_config_file, "r") as file:
        data_config = Box(yaml.safe_load(file))
    model_config_file = os.path.join(args.model_dir, "model_config.yaml")
    with open(model_config_file, "r") as file:
        model_config = Box(yaml.safe_load(file))
    sparse_dict_model = SparseDict(
        embedding_size=model_config.embedding_size,
        overcomplete_basis_size=model_config.overcomplete_basis_size,
    ).to(device)
    model_file = os.path.join(args.model_dir, "model_M.pth")
    model_dict = torch.load(model_file)
    sparse_dict_model.load_state_dict(model_dict)
    x_test = load_test_data(args, data_config)
    x_test = torch.from_numpy(x_test).to(device)
    import ipdb

    ipdb.set_trace()
    x_hat_test, c_test = sparse_dict_model(x_test)
    M = sparse_dict_model.encoder.weight.data
    import ipdb

    ipdb.set_trace()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("embedding_dir")
    parser.add_argument("model_dir")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    main(args, device)
