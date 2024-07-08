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
from psp.utils import DisentanglementScores
from sklearn.preprocessing import StandardScaler
import umap


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


def get_embeddings_for_label(label, all_embeddings, all_labels):
    all_labels = list(map(int, all_labels))
    labels_array = np.array(all_labels)
    mask = labels_array == label
    import ipdb

    ipdb.set_trace()
    return all_embeddings[mask]


def plot_embeddings(label, all_embeddings, all_labels):
    embeddings_for_label = get_embeddings_for_label(
        label, all_embeddings, all_labels
    )
    embs = umap.UMAP(random_state=42).fit(embeddings_for_label)
    plt.scatter(
        embs[:, 0],
        embs[:, 1],
        s=0.1,
        cmap="Spectral",
    )

    plt.savefig("clusterfck" + str(label) + ".png")


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

    plot_embeddings(1, delta_c_test.detach().cpu().numpy(), labels)
    import ipdb

    ipdb.set_trace()

    # get mcc score
    disentanglement_scores = DisentanglementScores()
    mcc_scores = disentanglement_scores.get_mcc_scores(
        delta_c_test.detach().cpu().numpy(),
        delta_z_test.detach().cpu().numpy(),
    )

    import ipdb

    ipdb.set_trace()

    # get comparative sparsities of transformations
    sp = []
    non_zero_sp = []
    for delta_c in delta_c_test:
        sp.append(
            torch.sum(torch.norm(delta_c, p=1)).detach().cpu().numpy().item()
        )
        non_zero_sp.append(
            np.sum(
                delta_c.detach().cpu().numpy() > float(args.sparsity_threshold)
            )
        )

    np_sp = np.array(sp)
    min_norm = np.min(np_sp)
    np_sp = np_sp / min_norm
    sparsity_penalties = list(np_sp)
    labels = list(map(int, labels))
    df = pd.DataFrame(
        {
            "Sparsity Penalties": sparsity_penalties,
            "Num-non-zeros": non_zero_sp,
            "Labels": labels,
        }
    )

    import ipdb

    ipdb.set_trace()

    # compare sparsities
    ax = sns.violinplot(
        x="Labels", y="Sparsity Penalties", data=df, fill=False
    )
    plt.title(
        "Variation of the L1 norm of reconstructed transformations with different p-sparse vectors"
    )
    medians = df.groupby(["Labels"])["Sparsity Penalties"].median()

    vertical_offset = df["Sparsity Penalties"].median() * 0.05

    for xtick, median_val in zip(ax.get_xticks(), medians):
        ax.text(
            xtick,
            median_val + vertical_offset,
            f"Median: {median_val:.2f}",
            horizontalalignment="center",
            size="x-small",
            color="black",
            weight="semibold",
        )
    plt.savefig("sparse_violins.png")

    import ipdb

    ipdb.set_trace()

    # get test error
    loss_fxn = torch.nn.MSELoss()
    test_losses = []
    for i in range(len(delta_z_test)):
        test_losses.append(
            loss_fxn(delta_z_hat_test[i], delta_z_test[i]).item()
        )
    plot(range(len(test_losses)), test_losses)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("embedding_dir")
    parser.add_argument("models_dir")
    parser.add_argument("--sparsity-threshold", default=float(1e-1))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    main(args, device)
