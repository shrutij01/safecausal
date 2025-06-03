import numpy as np
import argparse

import matplotlib.pyplot as plt
import matplotlib.ticker as tck
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
        values_pca = np.array([0.2893, 0.3660, 0.1613, 0.2763, 0.3202, 0.0307])
        std_devs_pca = np.array(
            [0.1200, 0.0481, 0.0586, 0.1389, 0.1251, 0.1014]
        )
        values_scope = np.array(
            [0.488, 0.5897, 0.58152, 0.4884, 0.4731, 0.6635]
        )
        std_devs_scope = np.array(
            [0.0578, 0.0703, 0.053, 0.0578, 0.07321, 0.0298]
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

        values_scope = np.array(
            [
                0.4155,
                0.3833,
            ]
        )
        std_devs_scope = np.array([0.0569, 0.1096])

        values_aff = np.array([0.5297, 0.4825])
        std_devs_aff = np.array([0.1268, 0.0731])

        values_md = np.array([0.4909, 0.4184])
        std_devs_md = np.array([0.1331, 0.1607])
        labels = [
            r"\textbf{\textsc{BINARY}(2, 2) $\rightarrow$ \textsc{LANG}(1, 1)}",
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
        values_scope,
        std_devs_scope,
        # values_pca,
        # std_devs_pca,
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
        values_scope,
        std_devs_scope,
        # values_pca,
        # std_devs_pca,
        labels,
        dataset_positions,
    ) = get_data(type=args.datatype)

    # sns.set_style("darkgrid")
    # # sns.set_context("paper")
    sns.set_context("notebook", rc={"lines.linewidth": 3})

    _, ax = plt.subplots(figsize=(21, 15))
    ax.xaxis.grid(False)
    width = 0.35
    ax.bar(
        dataset_positions - 2 * width,
        values_eta,
        label=r"$\mathbf{\tilde{z}_{\eta}}$",
        color="none",
        edgecolor="mediumslateblue",
        hatch="*",
        linewidth=4,
        width=0.4,
    )
    ax.errorbar(
        dataset_positions - 2 * width,
        values_eta,
        yerr=std_devs_eta,
        fmt="o",
        markersize=9,
        linestyle="",
        color="#36454F",
        alpha=0.5,
        elinewidth=3,
        capsize=9,
        capthick=3,
        # linestyle="--",
    )
    ax.bar(
        dataset_positions - width,
        values_scope,
        label=r"$\mathbf{\tilde{z}_{scope}}$",
        color="none",
        edgecolor="deepskyblue",
        hatch="..",
        linewidth=4,
        width=0.35,
    )
    ax.errorbar(
        dataset_positions - width,
        values_scope,
        yerr=std_devs_scope,
        fmt="o",
        markersize=9,
        linestyle="",
        color="#36454F",
        alpha=0.5,
        elinewidth=3,
        capsize=9,
        capthick=3,
        # linestyle="--",
    )
    ax.bar(
        dataset_positions,
        values_aff,
        label=r"$\mathbf{\tilde{z}_{aff}}$",
        color="none",
        edgecolor="lightseagreen",
        hatch="x",
        linewidth=4,
        width=0.35,
    )
    ax.errorbar(
        dataset_positions,
        values_aff,
        yerr=std_devs_aff,
        fmt="o",
        markersize=9,
        linestyle="",
        color="#36454F",
        alpha=0.5,
        elinewidth=3,
        capsize=9,
        capthick=3,
        # linestyle="--",
    )
    # ax.bar(
    #     dataset_positions + width,
    #     values_pca,
    #     label=r"$\mathbf{\tilde{z}_{PCA}}$",
    #     color="none",
    #     edgecolor="lightgreen",
    #     hatch="\\",
    #     capsize=9,
    #     linewidth=4,
    #     width=0.35,
    # )
    # ax.errorbar(
    #     dataset_positions + width,
    #     values_pca,
    #     yerr=std_devs_pca,
    #     fmt="o",
    #     markersize=9,
    #     linestyle="",
    #     color="#36454F",
    #     alpha=0.5,
    #     elinewidth=3,
    #     # elinestyle="--",
    #     capsize=9,
    #     capthick=3,
    #     # linestyle="--",
    # )
    ax.bar(
        dataset_positions + width,
        values_md,
        label=r"$\mathbf{\tilde{z}_{MD}}$",
        color="none",
        edgecolor="hotpink",
        hatch="x",
        linewidth=4,
        width=0.35,
    )
    ax.errorbar(
        dataset_positions + width,
        values_md,
        yerr=std_devs_md,
        fmt="o",
        markersize=9,
        linestyle="",
        color="#36454F",
        alpha=0.5,
        elinewidth=3,
        # elinestyle="--",
        capsize=9,
        capthick=3,
        # linestyle="--",
    )
    ax.set_xticks(dataset_positions)

    ax.set_xticklabels(
        labels=labels,
        size=33,
    )

    ax.set_ylim([0, 0.9])
    ax.set_title(
        r"\textbf{Cosine Similarity} $\mathbf{(\tilde{z}, \circ)}$ (Higher is better)",
        fontsize=45,
        pad=15,
    )
    # plt.yticks(np.arange(0, 0.9, 0.01))
    ax.yaxis.set_minor_locator(tck.AutoMinorLocator())
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
        size=33,
    )

    ax.legend(
        title=r"$\textbf{Steering Method} (\circ)$",
        title_fontsize=25,
        loc=2,
        ncol=5,
        prop={
            "size": 27,
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
            x=3.57, color="lightslategray", linestyle="--", linewidth=0.9
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
    ax.tick_params(labelsize=25)

    plt.setp(ax.get_xticklabels(), fontweight="bold")
    plt.setp(ax.get_yticklabels(), fontweight="bold")

    # Display the plot
    plt.savefig("varyingk_mccs.png")


# def plot_sensititivities():


def plot_udr():
    # [binary11] 0.01, 0.005, 0.007
    # udr_bin1_elev = np.array(
    #     [0.9865288369197924, 0.9933931773940217, 0.9908350564094865]
    # )
    # udr_bin1_fiv = np.array(
    #     [0.9925321906882463, 0.9947568255354127, 0.9936149051457699]
    # )
    # mcc_bin1_elev = np.array(
    #     [0.9867853798645869, 0.9933493251304302, 0.9910330195419018]
    # )
    # std_bin1_elev = np.array(
    #     [0.002513108443668445, 0.0009739848687034476, 0.0014317693843128936]
    # )
    # mcc_bin1_fiv = np.array(
    #     [0.9925907733211238, 0.9947732192663062, 0.9937015016758759]
    # )
    # std_bin1_fiv = np.array(
    #     [0.001528407397360336, 0.0010812111926471345, 0.0011634335698773976]
    # )
    # binary (2, 2)
    udr_bin2_elev = np.array(
        [0.9690717494531893, 0.9909487261786432, 0.9858168276715547]
    )
    mcc_bin2_elev = np.array(
        [0.9685572448992008, 0.9907978622655473, 0.9850805922898378]
    )
    std_bin2_elev = np.array(
        [0.0034486730605266984, 0.0009763470499424323, 0.001642862109428793]
    )
    udr_bin2_thir = np.array(
        [0.964901856859058, 0.9897532239171332, 0.983048094993364]
    )
    mcc_bin2_thir = np.array(
        [0.9642387291794329, 0.9898931980716276, 0.9832268791598769]
    )
    std_bin2_thir = np.array(
        [0.003173651511661824, 0.0014572809893865547, 0.0011190291779732504]
    )
    udr_bin2_fif = np.array(
        [0.9649018506761806, 0.9902402562139567, 0.9829261639724016]
    )
    mcc_bin2_fif = np.array(
        [0.9642387242706535, 0.989969570673862, 0.9831218310090705]
    )
    std_bin2_fif = np.array(
        [0.0031736497996879406, 0.0014121594868188418, 0.0011008048948131064]
    )

    indices = np.array([1, 2, 3])
    indices_2 = np.array([4, 5, 6])
    indices_3 = np.array([7, 8, 9])
    # all_indices = list(indices) + list(indices_2)
    all_indices = list(indices) + list(indices_2) + list(indices_3)
    # Plotting
    sns.set_context("notebook", rc={"lines.linewidth": 3})
    _, ax = plt.subplots(figsize=(21, 9))
    ax.yaxis.set_minor_locator(tck.AutoMinorLocator())
    # bin1
    # ax.errorbar(
    #     indices,
    #     mcc_bin1_elev,
    #     yerr=std_bin1_elev,
    #     fmt="-s",
    #     color="purple",
    #     capsize=5,
    #     linewidth=3,
    # )
    # ax.plot(
    #     indices,
    #     udr_bin1_elev,
    #     color="orange",
    #     marker="o",
    #     linewidth=5,
    #     markersize=10,
    # )
    # ax.errorbar(
    #     indices_2,
    #     mcc_bin1_fiv,
    #     yerr=std_bin1_fiv,
    #     label=r"\textbf{MCC}",
    #     fmt="-s",
    #     color="purple",
    #     capsize=5,
    #     linewidth=3,
    # )
    # ax.plot(
    #     indices_2,
    #     udr_bin1_fiv,
    #     label=r"\textbf{UDR}",
    #     color="orange",
    #     marker="o",
    #     markersize=10,
    #     linewidth=5,
    # )
    # bin2
    ax.errorbar(
        indices,
        mcc_bin2_elev,
        yerr=std_bin2_elev,
        fmt="-s",
        color="purple",
        capsize=5,
        linewidth=3,
    )
    ax.plot(
        indices,
        udr_bin2_elev,
        color="orange",
        marker="o",
        linewidth=5,
        markersize=10,
    )
    ax.errorbar(
        indices_2,
        mcc_bin2_thir,
        yerr=std_bin2_thir,
        label="MCC",
        fmt="-s",
        color="purple",
        capsize=5,
        linewidth=3,
    )
    ax.plot(
        indices_2,
        udr_bin2_thir,
        label="UDR",
        color="orange",
        marker="o",
        linewidth=5,
        markersize=10,
    )
    ax.errorbar(
        indices_3,
        mcc_bin2_fif,
        yerr=std_bin2_fif,
        # label="MCC",
        fmt="-s",
        color="purple",
        capsize=5,
        linewidth=3,
    )
    ax.plot(
        indices_3,
        udr_bin2_fif,
        # label="UDR",
        color="orange",
        marker="o",
        linewidth=5,
        markersize=10,
    )
    # Customizing the plot

    ax.set_xlabel(
        r"{$\texttt{primal\_lr}, \beta$}", fontweight="bold", fontsize=33
    )  # Using LaTeX for x-label
    ax.set_ylabel(r"\textbf{UDR/MCC}", fontweight="bold", fontsize=33)
    ax.set_title(
        r"\textbf{Comparison of UDR and MCC Values for} \textsc{Binary}(2, 2)",
        fontsize=30,
    )
    ax.set_xticks(all_indices)
    # ax.set_xticklabels(
    #     [
    #         r"\textbf{(0.01, 0.01)}",
    #         r"\textbf{(0.005, 0.01)}",
    #         r"\textbf{(0.007, 0.01)}",
    #         r"\textbf{(0.01, 0.1)}",
    #         r"\textbf{(0.005, 0.1)}",
    #         r"\textbf{(0.007, 0.1)}",
    #     ],
    #     fontsize=25,
    # )
    ax.set_xticklabels(
        [
            r"\textbf{(0.01, 0.2)}",
            r"\textbf{(0.005, 0.2)}",
            r"\textbf{(0.007, 0.2)}",
            r"\textbf{(0.01, 0.1)}",
            r"\textbf{(0.005, 0.1)}",
            r"\textbf{(0.007, 0.1)}",
            r"\textbf{(0.01, 0.05)}",
            r"\textbf{(0.005, 0.05)}",
            r"\textbf{(0.007, 0.05)}",
        ],
        fontsize=25,
    )
    ax.annotate(
        "Optimal HPs",
        xy=(all_indices[1], udr_bin2_elev[1]),
        xytext=(all_indices[4] + 15, udr_bin2_elev[1] - 75),
        arrowprops=dict(facecolor="black", shrink=0.05),
        fontsize=15,
        textcoords="offset points",
        ha="right",
    )
    plt.legend(
        loc=4,
        prop={
            "size": 25,
            "weight": "bold",
        },
    )
    ax.tick_params(labelsize=21)

    plt.setp(ax.get_xticklabels(), fontweight="bold")
    plt.setp(ax.get_yticklabels(), fontweight="bold")
    ax.grid(True)

    # Show plot
    # plt.show()
    plt.savefig("udr_mcc_bin1.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("datatype")
    parser.add_argument("--save", action="store_true")
    args = parser.parse_args()
    # main(args)
    # plot_varying_k_mccs()
    plot_udr()
