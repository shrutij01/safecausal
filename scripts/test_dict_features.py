import torch
from psp.sparse_dict import LinearAutoencoder


import argparse
import os
import yaml
from box import Box
import numpy as np
import pandas as pd

from terminalplot import plot
import seaborn as sns
import matplotlib.pyplot as plt
import psp.data_utils as data_utils
from psp.data_utils import tensorify, numpify

import psp.plot as psp_plot
from psp.evaluate import Evaluator


def prepare_test_data(args, device):
    data_config_file = os.path.join(args.embedding_dir, "config.yaml")
    with open(data_config_file, "r") as file:
        data_config = Box(yaml.safe_load(file))
    model_config_file = os.path.join(args.model_dir, "models_config.yaml")
    with open(model_config_file, "r") as file:
        model_config = Box(yaml.safe_load(file))
    sparse_dict_model = LinearAutoencoder(
        embedding_size=model_config.embedding_size,
        overcomplete_basis_size=model_config.overcomplete_basis_size,
    ).to(device)
    sparse_dict_model_file = os.path.join(
        args.model_dir, "sparse_dict_model", "sparse_dict_model.pth"
    )
    sparse_dict_model_dict = torch.load(sparse_dict_model_file)
    sparse_dict_model.load_state_dict(sparse_dict_model_dict)
    x_test, tf_ids, num_tfs = data_utils.load_test_data(args, data_config)
    delta_z_test = None
    delta_c_test = None
    if data_config.dataset == "toy_translator":
        delta_c_test = x_test
        if model_config.data_type == "gt_ent":
            lin_ent_tf = np.load(
                os.path.join(args.model_dir, "lin_ent_tf.npy")
            )
            delta_z_test = np.array(
                [
                    lin_ent_tf @ delta_c_test[i]
                    for i in range(delta_c_test.shape[0])
                ]
            )
        elif model_config.data_type == "gt":
            raise ValueError(
                "Maybe you shouldn't be running this experiment, check!"
            )
    elif model_config.data_type == "gradeschooler":
        delta_z_test = x_test
    delta_z_test = tensorify(delta_z_test, device)
    delta_z_hat_test, delta_c_hat_test = sparse_dict_model(delta_z_test)
    delta_c_hat_test = numpify(delta_c_hat_test)
    delta_z_hat_test = numpify(delta_z_hat_test)
    delta_z_test = numpify(delta_z_test)
    w_d = numpify(sparse_dict_model.decoder.weight.data)
    return (
        delta_c_test,
        delta_z_test,
        delta_c_hat_test,
        delta_z_hat_test,
        lin_ent_tf,
        w_d,
        num_tfs,
        tf_ids,
    )


def main(args, device):
    (
        delta_c_test,
        delta_z_test,
        delta_c_hat_test,
        delta_z_hat_test,
        lin_ent_tf,
        w_d,
        num_tfs,
        tf_ids,
    ) = prepare_test_data(args, device)

    evaluator = Evaluator(
        delta_c_test=delta_c_test,
        delta_z_test=delta_z_test,
        delta_c_hat_test=delta_c_hat_test,
        delta_z_hat_test=delta_z_hat_test,
        w_d_gt=lin_ent_tf,
        w_d=w_d,
        num_tfs=np.array(num_tfs),
        tf_ids=np.array(tf_ids),
    )
    evaluator.run_evaluations(plot=False)
    evaluator.get_metric_bounds()
    import ipdb

    ipdb.set_trace()

    psp_plot.plot_embeddings(None, delta_c_test, num_tfs, tf_ids, args)
    import ipdb

    ipdb.set_trace()

    threshold = float(1e-2)
    rounded_w_d = np.where(w_d < threshold, 0, w_d)
    print(rounded_w_d)

    import ipdb

    ipdb.set_trace()

    # get mcc score
    reps1, reps2 = data_utils.get_rep_pairs(1000, delta_c_test)

    import ipdb

    ipdb.set_trace()

    # get comparative sparsities of transformations
    sp = []
    non_zero_sp = []
    for delta_c in delta_c_test:
        sp.append(numpify(torch.sum(torch.norm(delta_c, p=1))).item())
        non_zero_sp.append(
            np.sum(numpify(delta_c) > float(args.sparsity_threshold))
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
    parser.add_argument("model_dir")
    parser.add_argument("--perplexity", default=5.0)
    parser.add_argument("--sparsity-threshold", default=float(5e-2))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    main(args, device)
