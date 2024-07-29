import torch
from psp.sparse_dict import LinearInvertible


import argparse
import os
import h5py
import yaml
from box import Box
import numpy as np
import pandas as pd
import re
import ast
from terminalplot import plot
import seaborn as sns
import matplotlib.pyplot as plt
from psp.utils import DisentanglementScores
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE


def load_test_data(args, data_config):
    delta_z = None
    tf_ids = []
    num_tfs = []
    df_file = os.path.join(args.embedding_dir, "object_translations1.csv")
    cfc_columns = data_config.cfc_column_names
    converters = {col: ast.literal_eval for col in cfc_columns}
    x_df = pd.read_csv(df_file, converters=converters)
    x_df_test = x_df.iloc[int(data_config.split * data_config.size) :]

    def convert_to_list_of_ints(value):
        if isinstance(value, str):
            value = ast.literal_eval(value)
        return [int(delta_z) for delta_z in value]

    for column in x_df_test[cfc_columns]:
        x_df_test[column] = x_df_test[column].apply(convert_to_list_of_ints)

    delta_z = np.asarray(
        (
            x_df_test[cfc_columns]
            .apply(lambda row: sum(row, []), axis=1)
            .tolist()
        )
    )
    num_tfs = list(map(int, num_tfs))
    return delta_z, tf_ids, num_tfs


def get_embeddings_for_num_tfs(
    target_num_tfs, all_embeddings, all_num_tfs, all_tf_ids
):
    all_num_tfs_array = np.array(all_num_tfs)
    mask_num_tfs = all_num_tfs_array == target_num_tfs
    label_names = np.array(all_tf_ids)[mask_num_tfs]
    return all_embeddings[mask_num_tfs], label_names


def plot_embeddings(
    target_num_tfs, all_embeddings, all_num_tfs, all_tf_ids, args
):
    if target_num_tfs is None:
        embeddings_for_label = all_embeddings
        label_names = all_num_tfs
        label = "all"
    else:
        embeddings_for_label, label_names = get_embeddings_for_num_tfs(
            target_num_tfs, all_embeddings, all_num_tfs, all_tf_ids
        )
        label = target_num_tfs
    if target_num_tfs == 1:
        legend_names = ["attribute", "color", "object"]
        perplexity = 3.0  # 5 is good too
    elif target_num_tfs == 2:
        legend_names = ["a+c", "c+o", "o+a"]
        perplexity = 13.0
    elif target_num_tfs is None:
        legend_names = ["one_tf", "two_tfs", "three_tfs"]
        perplexity = 25.0  # this is fuck all
    tsne = TSNE(
        random_state=1, metric="cosine", perplexity=float(args.perplexity)
    )
    embs = tsne.fit_transform(embeddings_for_label)

    unique_labels = sorted(set(label_names))
    cmap = plt.get_cmap("viridis")
    colors = cmap(np.linspace(0, 1, len(unique_labels)))
    label_to_color = dict(zip(unique_labels, colors))
    fig, ax = plt.subplots()
    scatter = ax.scatter(
        embs[:, 0],
        embs[:, 1],
        alpha=0.1,
        c=[label_to_color[name] for name in label_names],
    )
    legend_handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=color,
            markersize=10,
        )
        for color in colors
    ]
    ax.legend(legend_handles, legend_names, title="Legend")
    plt.savefig("clusterfck" + str(label) + str(args.perplexity) + ".png")


def get_rep_pairs(num_pairs, delta_c_test):
    total_samples = delta_c_test.shape[0]
    m = delta_c_test.shape[1]
    reps1 = np.zeros((num_pairs, m))
    reps2 = np.zeros((num_pairs, m))
    for i in range(num_pairs):
        id1, id2 = np.random.choice(total_samples, size=2, replace=False)

        reps1[i] = delta_c_test[id1]
        reps2[i] = delta_c_test[id2]
    return reps1, reps2


def main(args, device):
    data_config_file = os.path.join(args.embedding_dir, "config.yaml")
    with open(data_config_file, "r") as file:
        data_config = Box(yaml.safe_load(file))
    models_config_file = os.path.join(args.models_dir, "models_config.yaml")
    with open(models_config_file, "r") as file:
        models_config = Box(yaml.safe_load(file))
    sparse_dict_model = LinearInvertible(
        embedding_size=models_config.embedding_size,
        overcomplete_basis_size=models_config.overcomplete_basis_size,
    ).to(device)
    sparse_dict_model_file = os.path.join(
        args.models_dir, "sparse_dict_model", "sparse_dict_model.pth"
    )
    sparse_dict_model_dict = torch.load(sparse_dict_model_file)
    sparse_dict_model.load_state_dict(sparse_dict_model_dict)
    delta_z_test, tf_ids, num_tfs = load_test_data(args, data_config)
    delta_z_test = (
        torch.from_numpy(delta_z_test).to(device).type(torch.float32)
    )
    delta_z_hat_test, delta_c_test = sparse_dict_model(delta_z_test)
    delta_c_test = delta_c_test.detach().cpu().numpy()
    delta_z_hat_test = delta_z_hat_test.detach().cpu().numpy()
    delta_z_test = delta_z_test.detach().cpu().numpy()

    plot_embeddings(None, delta_c_test, num_tfs, tf_ids, args)
    import ipdb

    ipdb.set_trace()

    # get mcc score
    reps1, reps2 = get_rep_pairs(1000, delta_c_test)
    disentanglement_scores = DisentanglementScores()

    mcc_score = disentanglement_scores.get_mcc_score(
        reps1,
        reps2,
    )
    print(f"MCC on the test set is: {mcc_score}")

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
            "Labels": num_labels,
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
    parser.add_argument("--perplexity", default=5.0)
    parser.add_argument("--sparsity-threshold", default=float(5e-2))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    main(args, device)
