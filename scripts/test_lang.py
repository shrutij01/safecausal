from psp.linear_sae import LinearSAE
import psp.data_utils as utils
import psp.metrics as metrics

import argparse
import yaml
from box import Box

import os
import numpy as np
import itertools

from scipy import spatial
from sklearn.metrics.pairwise import cosine_similarity

import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def load_baseline(modeldir, dataconfig):
    model = LinearSAE(
        rep_dim=dataconfig.rep_dim,
        num_concepts=dataconfig.num_concepts,
        norm_type="bn",
    ).to(device)
    model.load_state_dict(
        torch.load(os.path.join(modeldir, "sparse_dict_model.pth"))
    )
    model.eval()
    return (
        model,
        utils.numpify(model.decoder.weight.data),
    )


def load_model(modeldir, dataconfig):
    # with open(
    #     os.path.join(modeldir, "prebias/", "model_config.yaml"),
    #     "r",
    # ) as file:
    #     modelconfig = Box(yaml.safe_load(file))
    with open(
        os.path.join(modeldir, "model_config.yaml"),
        "r",
    ) as file:
        modelconfig = Box(yaml.safe_load(file))
    model = LinearSAE(
        rep_dim=dataconfig.rep_dim,
        num_concepts=dataconfig.num_concepts,
        norm_type="bn",
    ).to(device)
    # model.load_state_dict(
    #     # torch.load(os.path.join(modeldir, "prebias/", "sparse_dict_model.pth"))
    # )
    model.load_state_dict(
        torch.load(os.path.join(modeldir, "sparse_dict_model.pth"))
    )
    model.eval()
    # model_string = str(modelconfig.alpha) + "_" + str(modelconfig.primal_lr)
    return (
        model,
        utils.numpify(model.decoder.weight.data),
        int(modelconfig.seed),
        # model_string,
    )


def compute_mccs(seeds, wds):
    pairs = list(itertools.combinations(range(len(seeds)), 2))
    mccs = []
    for pair in pairs:
        mccs.append(
            metrics.mean_corr_coef(
                wds[int(pair[0])],
                wds[int(pair[1])],
                method="pearson",
            ),
        )
    return mccs


def main(args):
    tilde_z, z = utils.load_test_data(
        data_file=args.data_file,
    )
    with open(args.dataconfig_file, "r") as file:
        data_config = Box(yaml.safe_load(file))
    modelrootdir = "/network/scratch/j/joshi.shruti/psp/binary_1_2/"
    modeldirnames = [
        args.modeldir_1,
        args.modeldir_2,
        # args.modeldir_3,
        # args.modeldir_4,
        # args.modeldir_5,
    ]

    modeldirs = [
        modelrootdir + str(modeldirname) for modeldirname in modeldirnames
    ]
    models, wds, seeds = [], [], []
    for modeldir in modeldirs:
        # model, wd, seed, model_string = load_model(modeldir, data_config)
        model, wd, seed = load_model(modeldir, data_config)
        models.append(model)
        wds.append(wd)
        seeds.append(seed)
    _, concept_projections = models[0](utils.tensorify((tilde_z - z), device))
    neta = concept_projections.detach().cpu().numpy() @ wds[0].T
    if (
        data_config.dataset == "binary_1"
        or data_config.dataset == "binary_1_2"
    ):
        mccs = compute_mccs(seeds, wds)
        mean_mcc = np.mean(mccs, axis=0)
        std_mcc = np.std(mccs, axis=0)
        model_string = "ok"
        mcc_string = "mccs_" + str(model_string)
        mean_mcc_string = "mean_mcc_" + str(model_string)
        std_mcc_string = "std_mcc_" + str(model_string)
        config_dict = {
            mcc_string: mccs,
            mean_mcc_string: mean_mcc,
            std_mcc_string: std_mcc,
        }
        print(config_dict)
        import ipdb

        ipdb.set_trace()
        md = utils.get_md_steering_vector(args.data_file)
        baseline, baseline_wd = load_baseline(args.baseline, data_config)
        _, baseline_concept_projections = baseline(
            utils.tensorify((tilde_z - z), device)
        )
        baseline_neta = (
            baseline_concept_projections.detach().cpu().numpy() @ baseline_wd.T
        )
        z = z / np.linalg.norm(z)
        z_md = z + md

        neta = neta / np.linalg.norm(neta)
        z_neta = z + neta
        z_neta = z_neta / np.linalg.norm(z_neta)
        z_md = z_md / np.linalg.norm(z_md)
        baseline_neta = baseline_neta / np.linalg.norm(baseline_neta)
        z_aff = z + baseline_neta
        z_aff = z_aff / np.linalg.norm(z_aff)
        tilde_z = tilde_z / np.linalg.norm(tilde_z)
        cosines_md, cosines_neta, cosines_aff = [], [], []
        for i in range(tilde_z.shape[0]):
            cosines_md.append(
                cosine_similarity(
                    tilde_z[i].reshape(1, -1), z_md[i].reshape(1, -1)
                )
            )
            cosines_neta.append(
                cosine_similarity(
                    tilde_z[i].reshape(1, -1), z_neta[i].reshape(1, -1)
                )
            )
            cosines_aff.append(
                cosine_similarity(
                    tilde_z[i].reshape(1, -1), z_aff[i].reshape(1, -1)
                )
            )
        plt.figure(figsize=(10, 6))
        cosines_md = [float(arr[0][0]) for arr in cosines_md]
        cosines_neta = [float(arr[0][0]) for arr in cosines_neta]
        cosines_aff = [float(arr[0][0]) for arr in cosines_aff]
        sns.kdeplot(
            cosines_md,
            bw_adjust=0.75,
            label=r"$\theta(\tilde{z}, \tilde{z}_{\text{MD}})$",
            shade=True,
            linewidths=1.5,
        )
        sns.kdeplot(
            cosines_neta,
            bw_adjust=0.75,
            label=r"$\theta(\tilde{z}, \tilde{z}_{\text{neta}})$",
            shade=True,
            linewidths=1.5,
        )
        sns.kdeplot(
            cosines_aff,
            bw_adjust=0.75,
            label=r"$\theta(\tilde{z}, \tilde{z}_{\text{aff}})$",
            shade=True,
            linewidths=1.5,
        )
        # plt.title("Cosine Similarities")
        ax = plt.gca()  # Get current axis
        ax.spines["top"].set_linewidth(1.5)
        ax.spines["right"].set_linewidth(1.5)
        ax.spines["left"].set_linewidth(1.5)
        ax.spines["bottom"].set_linewidth(1.5)
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)

        # Setting font sizes for the plot elements
        ax.tick_params(axis="both", which="major", labelsize=13)  # Ticks
        plt.xlabel("Cosine Similarity", fontsize=15)  # X-axis label
        plt.ylabel("Density", fontsize=15)  # Y-axis label
        # plt.title("Cosine Similarities")
        plt.legend(loc=2, prop={"size": 15})
        plt.savefig("kde____" + str(data_config.dataset) + "_" + ".png")
        import ipdb

        ipdb.set_trace()
        data = np.vstack([tilde_z, z_md, z_neta])

        # Create labels for each set
        labels = np.array(
            ["tilde_z"] * tilde_z.shape[0]
            + ["z_md"] * z_md.shape[0]
            + ["z_neta"] * z_neta.shape[0]
        )

        # Step 2: Apply t-SNE
        tsne = TSNE(n_components=2, perplexity=2, random_state=42)
        transformed_data = tsne.fit_transform(data)

        # Step 3: Plot the results
        plt.figure(figsize=(10, 8))
        sns.scatterplot(
            x=transformed_data[:, 0],
            y=transformed_data[:, 1],
            hue=labels,
            style=labels,
            palette="viridis",
            s=100,
        )
        plt.title("t-SNE visualization of tilde_z, z_md, z_neta")
        plt.xlabel("t-SNE 1")
        plt.ylabel("t-SNE 2")
        plt.legend(title="Steering")
        plt.grid(True)
        plt.savefig("tsne_" + str(data_config.dataset) + "_" + ".png")
    elif data_config.dataset == "binary_2":
        compute_mccs(seeds, wds)
        neta = neta / np.linalg.norm(neta)
        z = z / np.linalg.norm(z)
        z_neta = z + neta
        z_neta = z_neta / np.linalg.norm(z_neta)
        tilde_z = tilde_z / np.linalg.norm(tilde_z)
        cosines_neta = []
        for i in range(tilde_z.shape[0]):
            cosines_neta.append(
                cosine_similarity(
                    tilde_z[i].reshape(1, -1), z_neta[i].reshape(1, -1)
                )
            )
        cosines_neta = [float(arr[0][0]) for arr in cosines_neta]
        plt.figure(figsize=(10, 6))
        sns.kdeplot(
            cosines_neta,
            bw_adjust=0.75,
            label="Cosine Similarity with neta",
            shade=True,
        )
        plt.title("KDE of Cosine Similarities")
        plt.xlabel("Cosine Similarity")
        plt.ylabel("Density")
        plt.legend()
        plt.savefig("kde_" + str(data_config.dataset) + "_" + ".png")
        data = np.vstack([tilde_z, z_neta])

        # Create labels for each set
        labels = np.array(
            ["tilde_z"] * tilde_z.shape[0] + ["z_neta"] * z_neta.shape[0]
        )

        # Step 2: Apply t-SNE
        tsne = TSNE(n_components=2, perplexity=5, random_state=42)
        transformed_data = tsne.fit_transform(data)

        # Step 3: Plot the results
        plt.figure(figsize=(10, 8))
        sns.scatterplot(
            x=transformed_data[:, 0],
            y=transformed_data[:, 1],
            hue=labels,
            style=labels,
            palette="viridis",
            s=100,
        )
        plt.title("t-SNE visualization of tilde_z, z_neta")
        plt.xlabel("t-SNE 1")
        plt.ylabel("t-SNE 2")
        plt.legend(title="Steering")
        plt.grid(True)
        plt.savefig("tsne_" + str(data_config.dataset) + "_" + ".png")
        # test steering vectors on 1-sparse data
        onesp_tilde_z, onesp_z = utils.load_test_data(
            data_file=args.data_file2,
        )
        _, concept_projections_2_to1 = models[0](
            utils.tensorify((onesp_tilde_z - onesp_z), device)
        )
        neta_2_to1 = (concept_projections_2_to1.detach().cpu().numpy()) @ wds[
            0
        ].T
        neta_2_to1 = neta_2_to1 / np.linalg.norm(neta_2_to1)
        onesp_z = onesp_z / np.linalg.norm(onesp_z)
        z_neta_2_to_1 = onesp_z + neta_2_to1
        z_neta_2_to_1 = z_neta_2_to_1 / np.linalg.norm(z_neta_2_to_1)
        onesp_tilde_z = onesp_tilde_z / np.linalg.norm(onesp_tilde_z)
        cosines_neta_2_to1 = []
        for i in range(onesp_tilde_z.shape[0]):
            cosines_neta_2_to1.append(
                cosine_similarity(
                    onesp_tilde_z[i].reshape(1, -1),
                    z_neta_2_to_1[i].reshape(1, -1),
                )
            )
        cosines_neta_2_to1 = [float(arr[0][0]) for arr in cosines_neta_2_to1]
        md_2_to_1 = utils.get_md_steering_vector(args.data_file2)
        z_md_2_to_1 = onesp_z + md_2_to_1
        cosines_md_2_to_1 = []
        for i in range(onesp_tilde_z.shape[0]):
            cosines_md_2_to_1.append(
                cosine_similarity(
                    onesp_tilde_z[i].reshape(1, -1),
                    z_md_2_to_1[i].reshape(1, -1),
                )
            )
        cosines_md_2_to_1 = [float(arr[0][0]) for arr in cosines_md_2_to_1]
        plt.figure(figsize=(10, 6))
        sns.kdeplot(
            cosines_neta_2_to1,
            bw_adjust=0.75,
            label="Cosine Similarity with neta_2_to_1",
            shade=True,
        )
        sns.kdeplot(
            cosines_md_2_to_1,
            bw_adjust=0.75,
            label="Cosine Similarity with md_2_to_1",
            shade=True,
        )
        plt.title("KDE of Cosine Similarities")
        plt.xlabel("Cosine Similarity")
        plt.ylabel("Density")
        plt.legend()
        plt.savefig("kde_" + str(data_config.dataset) + "_2_to_1_2_" + ".png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("data_file")
    parser.add_argument("dataconfig_file")
    parser.add_argument("modeldir_1")
    parser.add_argument("modeldir_2")
    # parser.add_argument("modeldir_3")
    # parser.add_argument("modeldir_4")
    # parser.add_argument("modeldir_5")
    # parser.add_argument("baseline")
    parser.add_argument(
        "--data-file2",
        default="/network/scratch/j/joshi.shruti/psp/binary_1/binary_1_32_config.yaml",
    )
    args = parser.parse_args()
    main(args)
