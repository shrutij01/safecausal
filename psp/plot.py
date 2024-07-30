import numpy as np

import psp.data_utils as data_utils

from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE

import matplotlib.pyplot as plt
import seaborn as sns


def plot_embeddings(
    target_num_tfs, all_embeddings, all_num_tfs, all_tf_ids, args
):
    if target_num_tfs is None:
        embeddings_for_label = all_embeddings
        label_names = all_num_tfs
        label = "all"
    else:
        embeddings_for_label, label_names = (
            data_utils.get_embeddings_for_num_tfs(
                target_num_tfs, all_embeddings, all_num_tfs, all_tf_ids
            )
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


