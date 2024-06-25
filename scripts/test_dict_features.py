import torch
from psp.sparse_dict import SparseDict, AffineLayer

import argparse
import os
import h5py
import yaml
from box import Box
import numpy as np
import pandas as pd
import ast
from terminalplot import plot


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
    models_config_file = os.path.join(args.models_dir, "models_config.yaml")
    with open(models_config_file, "r") as file:
        models_config = Box(yaml.safe_load(file))
    sparse_dict_model = SparseDict(
        embedding_size=models_config.embedding_size,
        overcomplete_basis_size=models_config.overcomplete_basis_size,
    ).to(device)
    sparse_dict_model_file = os.path.join(
        args.models_dir, "sparse_dict_model", "sparse_dict_model.pth"
    )
    sparse_dict_model_dict = torch.load(sparse_dict_model_file)
    sparse_dict_model.load_state_dict(sparse_dict_model_dict)
    r_model = AffineLayer(embedding_size=models_config.embedding_size)
    r_model_file = os.path.join(args.models_dir, "r_model", "r_model.pth")
    r_model_dict = torch.load(r_model_file)
    r_model.load_state_dict(r_model_dict)
    delta_z_test = load_test_data(args, data_config)
    delta_z_test = (
        torch.from_numpy(delta_z_test).to(device).type(torch.float32)
    )
    delta_z_hat_test, delta_c_test = sparse_dict_model(delta_z_test)

    # get transformations
    print(delta_c_test)

    # compute dict features
    W_d = sparse_dict_model.decoder.weight.data.cpu()
    threshold = float(1e-5)
    rounded_W_d = np.where(W_d < threshold, 0, W_d)
    print(rounded_W_d)

    # get test error
    loss_fxn = torch.nn.MSELoss()
    test_losses = loss_fxn(delta_z_hat_test, delta_z_test)
    plot(range(len(test_losses)), test_losses)

    import ipdb

    ipdb.set_trace()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("embedding_dir")
    parser.add_argument("models_dir")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    main(args, device)
