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
from terminalplot import plot
import seaborn as sns
import matplotlib.pyplot as plt


def load_test_data(args, data_config):
    delta_z = None
    labels = []
    if data_config.dataset == "toy_translator":
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
            return [int(delta_z) for delta_z in value]

        for column in x_df_test[cfc_columns]:
            x_df_test[column] = x_df_test[column].apply(
                convert_to_list_of_ints
            )

        delta_z = np.asarray(
            (
                x_df_test[cfc_columns]
                .apply(lambda row: sum(row, []), axis=1)
                .tolist()
            )
        )
    elif data_config.dataset == "gradeschooler":
        embeddings_file = os.path.join(args.embedding_dir, "embeddings.h5")
        with h5py.File(embeddings_file, "r") as f:
            cfc1_embeddings_test = np.array(f["cfc1_test"])
            cfc2_embeddings_test = np.array(f["cfc2_test"])
        delta_z = cfc2_embeddings_test - cfc1_embeddings_test
        mean = delta_z.mean(axis=0)
        std = delta_z.std(axis=0, ddof=1)
        delta_z = (delta_z - mean) / (std + 1e-8)
        labels_file = os.path.join(args.embedding_dir, "gradeschooler.txt")
        labels = []
        with open(labels_file, "r") as f:
            context_pairs = [
                line.strip().split("\t") for line in f if line.strip()
            ]
        split = int(0.9 * data_config.dataset_length)
        for cp in context_pairs[split:]:
            labels.append(cp[0].split(",")[2])
    else:
        raise NotImplementedError(
            "Datasets implemented: toy_translator and gradeschooler"
        )
    return delta_z, labels


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
    delta_z_test, labels = load_test_data(args, data_config)
    delta_z_test = (
        torch.from_numpy(delta_z_test).to(device).type(torch.float32)
    )
    delta_z_hat_test, delta_c_test = sparse_dict_model(delta_z_test)

    import ipdb

    ipdb.set_trace()

    # get comparative sparsities of transformations
    sparsity_penalties = []
    for delta_c in delta_c_test:
        sparsity_penalties.append(
            torch.sum(torch.norm(delta_c, p=1)).detach().cpu().numpy().item()
        )
    labels = list(map(int, labels))
    df = pd.DataFrame(
        {"Sparsity Penalties": sparsity_penalties, "Labels": labels}
    )
    df["Sparsity Penalties"] = pd.to_numeric(
        df["Sparsity Penalties"], errors="coerce"
    )
    import ipdb

    ipdb.set_trace()
    sns.violinplot(x="Labels", y="Sparsity Penalties", data=df)
    plt.title(
        "Variation of the L1 norm of reconstructed transformations with different p-sparse vectors"
    )
    plt.savefig("sparse_violins.png")

    # compute dict features
    W_d = sparse_dict_model.decoder.weight.data.cpu()
    threshold = float(1e-5)
    rounded_W_d = np.where(W_d < threshold, 0, W_d)
    print(rounded_W_d)

    import ipdb

    ipdb.set_trace()

    # get test error
    loss_fxn = torch.nn.MSELoss()
    test_losses = []
    for _ in range(len(delta_z_test)):
        test_losses.append(loss_fxn(delta_z_hat_test, delta_z_test).item())
    plot(range(len(test_losses)), test_losses)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("embedding_dir")
    parser.add_argument("models_dir")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    main(args, device)
