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
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def load_model(modeldir, dataconfig):
    with open(os.path.join(modeldir, "model_config.yaml"), "r") as file:
        modelconfig = Box(yaml.safe_load(file))
    model = LinearSAE(
        rep_dim=dataconfig.rep_dim,
        num_concepts=dataconfig.num_concepts,
        norm_type=modelconfig.norm_type,
    ).to(device)
    model.load_state_dict(
        torch.load(os.path.join(modeldir, "sparse_dict_model.pth"))
    )
    return model, utils.numpify(model.decoder.weight.data), modelconfig.seed


def main(args):
    tilde_z, z = utils.load_test_data(
        data_file=args.data_file,
    )
    with open(args.dataconfig_file, "r") as file:
        data_config = Box(yaml.safe_load(file))
    modeldirs = [args.modeldir_1, args.modeldir_2, args.modeldir_3]
    models, wds, seeds = [], [], []
    for modeldir in modeldirs:
        model, wd, seed = load_model(modeldir, data_config)
        models.append(model)
        wds.append(wd)
        seeds.append(seed)
    if data_config.dataset == "binary_1":
        md = utils.get_md_steering_vector(args.data_file)
        cosine_similarities = []
        for i in range(3):
            cosine_similarities.append(
                1 - spatial.distance.cosine(wds[i].squeeze(), md)
            )
        print("cosine_similarities with md: ", cosine_similarities)

        pairs = list(itertools.combinations(seeds, 2))
        mccs = []
        for pair in pairs:
            mccs.append(
                metrics.mean_corr_coef(
                    wds[int(pair[0])],
                    wds[int(pair[1])],
                    method="pearson",
                ),
            )
        print("mccs: ", mccs)
        z_md = z + md
        z_neta = z + wds[0].squeeze()
        cosines_md, cosines_neta = [], []
        for i in range(tilde_z.shape[0]):
            cosines_md.append(1 - spatial.distance.cosine(tilde_z[i], z_md[i]))
            cosines_neta.append(
                1 - spatial.distance.cosine(tilde_z[i], z_neta[i])
            )
        plt.figure(figsize=(10, 6))

        sns.kdeplot(
            cosines_md,
            label="Cosine Similarity with MD",
            shade=True,
        )
        sns.kdeplot(
            cosines_neta,
            label="Cosine Similarity with neta",
            shade=True,
        )
        plt.title("KDE of Cosine Similarities")
        plt.xlabel("Cosine Similarity")
        plt.ylabel("Density")
        plt.legend()
        plt.savefig("kde_binary_1.png")
        import ipdb

        ipdb.set_trace()
        data = np.vstack([tilde_z, z_md, z_neta])

        # Create labels for each set
        labels = np.array(["tilde_z"] * 10 + ["z_md"] * 10 + ["z_neta"] * 10)

        # Step 2: Apply t-SNE
        tsne = TSNE(n_components=2, perplexity=10, random_state=42)
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
        plt.savefig("tsne_binary_1.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("data_file")
    parser.add_argument("dataconfig_file")
    parser.add_argument("modeldir_1")
    parser.add_argument("modeldir_2")
    parser.add_argument("modeldir_3")
    args = parser.parse_args()
    main(args)
