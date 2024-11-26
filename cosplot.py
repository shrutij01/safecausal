import numpy as np
import argparse

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import matplotlib.ticker as ticker


matplotlib.rcParams["text.usetex"] = True

matplotlib.rcParams["text.latex.preamble"] = r"\usepackage{amsmath}"
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
            r"$\textbf{\textsc{BINARY}(2, 2) \rightarrow \textsc{GENDER}(1, 1)}$",
            r"$\textbf{\textsc{CORR}(2, 1) \rightarrow \textsc{LANG}(1, 1)}$",
        ]

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
    dataset_positions = np.array([2 * i + 1 for i in range(len(labels))])
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

    sns.set_theme(style="whitegrid")
    _, ax = plt.subplots(figsize=(15, 7))

    width = 0.2
    ax.errorbar(
        dataset_positions - width,
        values_eta,
        yerr=std_devs_eta,
        fmt="o",
        markersize=9,
        label=r"$\mathbf{\tilde{z}_{\eta}}$",
        color="mediumslateblue",
        capsize=9,
        capthick=1.5,
        linewidth=3.5,
    )
    ax.errorbar(
        dataset_positions,
        values_aff,
        yerr=std_devs_aff,
        fmt="o",
        markersize=9,
        label=r"$\mathbf{\tilde{z}_{aff}}$",
        color="lightseagreen",
        capsize=9,
        capthick=1.5,
        linewidth=3.5,
    )
    ax.errorbar(
        dataset_positions + width,
        values_md,
        yerr=std_devs_md,
        fmt="o",
        markersize=9,
        label=r"$\mathbf{\tilde{z}_{MD}}$",
        color="hotpink",
        capsize=9,
        capthick=1.5,
        linewidth=3.5,
    )
    ax.set_xticks(dataset_positions)

    ax.set_xticklabels(
        labels=labels,
    )
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.set_xlim([0, 12])

    ax.set_ylim([0, 0.9])
    ax.set_ylabel(
        r"\textbf{Cosine Similarity} $\mathbf{\boldsymbol{\theta}(\tilde{z}, \circ)}$ (Higher is better)",
        fontsize=25,
        labelpad=20,
    )
    ax.set_yticklabels(
        labels=[
            r"\textbf{0.0}",
            r"\textbf{0.1}",
            r"\textbf{0.2}",
            r"\textbf{0.3}",
            r"\textbf{0.4}",
            r"\textbf{0.5}",
            r"\textbf{0.6}",
            r"\textbf{0.7}",
            r"\textbf{0.8}",
            r"\textbf{0.9}",
        ],
    )

    ax.legend(
        title=r"$\textbf{Steering Method} (\circ)$",
        title_fontsize=19,
        loc=2,
        prop={
            "size": 23,
            "weight": "bold",
        },
    )
    ax.grid(
        True,
        which="major",
        linestyle="-",
        linewidth=1.0,
        color="lightsteelblue",
    )
    for i in range(2, 2 * len(dataset_positions) - 1, 2):
        ax.axvline(x=i, color="gray", linestyle="-", linewidth=1.5)
    ax.tick_params(axis="both", which="major", labelsize=19)

    plt.tight_layout()
    if args.save:
        print("Saving the plot...")
        plt.savefig("cosine_sims_" + str(args.datatype) + "_.png")
    else:
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("datatype")
    parser.add_argument("--save", action="store_true")
    args = parser.parse_args()
    main(args)
