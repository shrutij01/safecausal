import numpy as np
import argparse

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

plt.rcParams.update(
    {
        "text.usetex": True,
        "text.latex.preamble": r"\usepackage{amsmath}",
    }
)
plt.rc("font", family="serif", size=19)


def get_data(type="same"):
    if type == "same":
        values_eta = np.array([0.4624, 0.6522, 0.7732, 0.6901, 0.7576, 0.7339])
        std_devs_eta = np.array(
            [0.0850, 0.0719, 0.0610, 0.0912, 0.0826, 0.0838]
        )

        values_aff = np.array([0.4263, 0.6074, 0.5819, 0.4417, 0.6100, 0.5963])
        std_devs_aff = np.array(
            [0.1598, 0.1296, 0.1018, 0.1343, 0.1014, 0.1327]
        )

        values_md = np.array([0.4184, 0.4909, np.nan, np.nan, np.nan, 0.1789])
        std_devs_md = np.array(
            [0.1607, 0.1331, np.nan, np.nan, np.nan, 0.1377]
        )
        labels = [
            r"$\textbf{\textsc{LANG}(1, 1)}$",
            r"$\textbf{\textsc{GENDER}(1, 1)}$",
            r"$\textbf{\textsc{BINARY}(2, 2)}$",
            r"$\textbf{\textsc{CORR}(2, 1)}$",
            r"$\textbf{\textsc{CAT}(135, 3)}$",
            r"$\textbf{\text{TruthfulQA}(1, 1)}$",
        ]
        dataset_positions = np.array([2 * i + 1 for i in range(len(labels))])
    elif type == "ood":
        values_eta = np.array(
            [
                0.7451,
                0.7134,
            ]
        )
        std_devs_eta = np.array([0.0719, 0.0965])

        values_aff = np.array([0.5297, 0.4825])
        std_devs_aff = np.array([0.1268, 0.0731])

        values_md = np.array([0.4909, 0.4184])
        std_devs_md = np.array([0.1331, 0.1607])
        labels = [
            r"\textbf{\textsc{BINARY}(2, 2) $\rightarrow$ \textsc{GENDER}(1, 1)}",
            r"\textbf{\textsc{CORR}(2, 1) $\rightarrow$ \textsc{LANG}(1, 1)}",
        ]
        dataset_positions = np.array([3, 4.5])

    else:
        raise ValueError("Invalid type")
    assert (
        len(labels)
        == len(std_devs_md)
        == len(std_devs_aff)
        == len(std_devs_eta)
        == len(values_md)
        == len(values_aff)
        == len(values_eta)
    )
    return (
        values_eta,
        std_devs_eta,
        values_aff,
        std_devs_aff,
        values_md,
        std_devs_md,
        labels,
        dataset_positions,
    )


def main(args):
    (
        values_eta,
        std_devs_eta,
        values_aff,
        std_devs_aff,
        values_md,
        std_devs_md,
        labels,
        dataset_positions,
    ) = get_data(type=args.datatype)

    sns.set_style("darkgrid")
    _, ax = plt.subplots(figsize=(15, 9))
    ax.xaxis.grid(False)
    width = 0.35
    ax.bar(
        dataset_positions - width,
        values_eta,
        label=r"$\mathbf{\tilde{z}_{\eta}}$",
        color="mediumslateblue",
        edgecolor="gray",
        linewidth=1.5,
        width=0.4,
    )
    ax.errorbar(
        dataset_positions - width,
        values_eta,
        yerr=std_devs_eta,
        fmt="o",
        markersize=9,
        linestyle="",
        color="black",
        elinewidth=3,
        capsize=9,
        capthick=3,
    )

    ax.bar(
        dataset_positions,
        values_aff,
        label=r"$\mathbf{\tilde{z}_{aff}}$",
        color="lightseagreen",
        edgecolor="gray",
        linewidth=1.5,
        width=0.4,
    )
    ax.errorbar(
        dataset_positions,
        values_aff,
        yerr=std_devs_aff,
        fmt="o",
        markersize=9,
        linestyle="",
        color="black",
        elinewidth=3,
        capsize=9,
        capthick=3,
    )
    ax.bar(
        dataset_positions + width,
        values_md,
        yerr=std_devs_md,
        label=r"$\mathbf{\tilde{z}_{MD}}$",
        color="hotpink",
        capsize=9,
        edgecolor="gray",
        linewidth=1.5,
        width=0.4,
    )
    ax.errorbar(
        dataset_positions + width,
        values_md,
        yerr=std_devs_md,
        fmt="o",
        markersize=9,
        linestyle="",
        color="black",
        elinewidth=3,
        capsize=9,
        capthick=3,
    )
    ax.set_xticks(dataset_positions)

    ax.set_xticklabels(
        labels=labels,
        size=23,
    )

    ax.set_ylim([0, 0.9])
    ax.set_title(
        r"\textbf{Cosine Similarity} $\mathbf{\boldsymbol{\theta}(\tilde{z}, \circ)}$ (Higher is better)",
        fontsize=33,
        pad=15,
    )
    ax.set_yticklabels(
        labels=[
            r"$\textbf{0.0}$",
            r"$\textbf{0.1}$",
            r"$\textbf{0.2}$",
            r"$\textbf{0.3}$",
            r"$\textbf{0.4}$",
            r"$\textbf{0.5}$",
            r"$\textbf{0.6}$",
            r"$\textbf{0.7}$",
            r"$\textbf{0.8}$",
            r"$\textbf{0.9}$",
        ],
        size=23,
    )

    ax.legend(
        title=r"$\textbf{Steering Method} (\circ)$",
        title_fontsize=23,
        loc=1,
        prop={
            "size": 25,
            "weight": "bold",
        },
    )
    if args.datatype == "same":
        for i in range(2, 2 * len(dataset_positions) - 1, 2):
            ax.axvline(
                x=i, color="lightslategray", linestyle="--", linewidth=0.9
            )
    elif args.datatype == "ood":
        ax.axvline(
            x=3.75, color="lightslategray", linestyle="--", linewidth=0.9
        )

    plt.tight_layout()
    if args.save:
        print("Saving the plot...")
        plt.savefig("cosine_sims_" + str(args.datatype) + "_.png")
    else:
        plt.show()


def plot_varying_k_mccs():

    mean_mccs_cat_eta = np.array(
        [
            0.9059,
            0.8843,
            0.8613,
            0.8398,
            0.8236,
            0.8129,
            0.7549,
        ]
    )
    std_dev_cat_eta = np.array(
        [0.02186, 0.0223, 0.0318, 0.0313, 0.0334, 0.0383, 0.0391]
    )
    k = np.array([135, 150, 200, 250, 300, 350, 500])
    mean_mccs_cat_aff = np.array(
        [
            0.6607,
            0.5987,
            0.5353,
            0.4892,
            0.4681,
            0.3473,
            0.3226,
        ]
    )
    std_dev_cat_aff = np.array(
        [0.0189, 0.0214, 0.0205, 0.0205, 0.0217, 0.0228, 0.0213]
    )
    sns.set_style("darkgrid")
    _, ax = plt.subplots(figsize=(10, 6))

    # Plot the first line with shaded standard deviation
    plt.plot(
        k,
        mean_mccs_cat_eta,
        label=r"$\eta$",
        color="mediumslateblue",
        linewidth=3,
    )
    plt.fill_between(
        k,
        mean_mccs_cat_eta - std_dev_cat_eta,
        mean_mccs_cat_eta + std_dev_cat_eta,
        color="mediumslateblue",
        alpha=0.2,
    )

    # Plot the second line with shaded standard deviation
    plt.plot(
        k,
        mean_mccs_cat_aff,
        label=r"$\text{aff}$",
        color="lightseagreen",
        linewidth=3,
    )
    plt.fill_between(
        k,
        mean_mccs_cat_aff - std_dev_cat_aff,
        mean_mccs_cat_aff + std_dev_cat_aff,
        color="lightseagreen",
        alpha=0.2,
    )

    # Add labels and legend
    plt.xlabel(r"\textbf{k}", fontweight="bold")
    plt.ylabel(r"\textbf{Mean MCC}", fontweight="bold")
    ax.set_title(
        r"$\textbf{\textsc{CAT}}(135, 3): \text{MCCs for Increasing} \hspace{2mm} k$",
        fontsize=33,
        pad=15,
    )
    plt.legend(
        title=r"$\textbf{Method}$",
        title_fontsize=23,
        loc=3,
        prop={
            "size": 25,
            "weight": "bold",
        },
    )
    ax.tick_params(labelsize=23)

    plt.setp(ax.get_xticklabels(), fontweight="bold")
    plt.setp(ax.get_yticklabels(), fontweight="bold")

    # Display the plot
    plt.savefig("varyingk_mccs.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("datatype")
    parser.add_argument("--save", action="store_true")
    args = parser.parse_args()
    # main(args)
    plot_varying_k_mccs()
