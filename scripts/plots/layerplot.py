import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import matplotlib.ticker as tck

plt.rcParams.update(
    {
        "text.usetex": True,
        "text.latex.preamble": r"\usepackage{amsmath}",
    }
)
plt.rc("font", family="serif", size=19, weight="bold")

ef_cos_mean = [0.5915, 0.5590, 0.4526, 0.5071, 0.5088, 0.2724, 0.2067, 0.1778]
ef_cos_std = [0.1190, 0.1149, 0.1442, 0.1033, 0.1063, 0.0698, 0.0812, 0.0418]

corr_cos_mean_ef = [
    0.5385,
    0.5359,
    0.3034,
    0.4343,
    0.4577,
    0.2628,
    0.1646,
    0.1480,
]
corr_cos_std_ef = [
    0.1047,
    0.1138,
    0.1154,
    0.0782,
    0.0908,
    0.0746,
    0.0716,
    0.0304,
]

corr_cos_mean_eg = [
    0.54133,
    0.54637,
    0.3086,
    0.4407,
    0.4691,
    0.2741,
    0.1691,
    0.1312,
]
corr_cos_std_eg = [
    0.13438,
    0.15502,
    0.1470,
    0.0849,
    0.0912,
    0.0827,
    0.0759,
    0.0791,
]
sns.despine(right=True)

labels = ["32", "31", "21", "15", "13", "5", "3", "1"]
x = np.arange(len(labels))

_, ax = plt.subplots(figsize=(15, 11))

# Plot each line with its standard deviation as a filled region
plt.plot(
    x,
    ef_cos_mean,
    label=r"\textbf{LANG(1,1)}, \texttt{eng} $\rightarrow$ \texttt{french}",
    color="#8e44ad",
    marker="o",
    markersize=15,
    linewidth=7,
)
plt.fill_between(
    x,
    np.array(ef_cos_mean) - np.array(ef_cos_std),
    np.array(ef_cos_mean) + np.array(ef_cos_std),
    color="#8e44ad",
    alpha=0.1,
)

plt.plot(
    x,
    corr_cos_mean_ef,
    label=r"\textbf{CORR(2,1)}, \texttt{eng} $\rightarrow$ \texttt{french}",
    color="#2980b9",
    marker="o",
    markersize=15,
    linewidth=7,
)
plt.fill_between(
    x,
    np.array(corr_cos_mean_ef) - np.array(corr_cos_std_ef),
    np.array(corr_cos_mean_ef) + np.array(corr_cos_std_ef),
    color="#2980b9",
    alpha=0.1,
)

plt.plot(
    x,
    corr_cos_mean_eg,
    label=r"\textbf{CORR(2,1)}, \texttt{eng} $\rightarrow$ \texttt{german}",
    color="#27ae60",
    marker="o",
    markersize=15,
    linewidth=7,
)
plt.fill_between(
    x,
    np.array(corr_cos_mean_eg) - np.array(corr_cos_std_eg),
    np.array(corr_cos_mean_eg) + np.array(corr_cos_std_eg),
    color="#27ae60",
    alpha=0.1,
)
ax.yaxis.set_minor_locator(tck.AutoMinorLocator())
plt.xticks(x, labels, fontsize=25, fontweight="bold")

plt.yticks(fontsize=25, fontweight="bold")
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
    size=35,
)
ax.set_xticklabels(
    labels=[
        r"$\textbf{32}$",
        r"$\textbf{31}$",
        r"$\textbf{21}$",
        r"$\textbf{15}$",
        r"$\textbf{13}$",
        r"$\textbf{5}$",
        r"$\textbf{3}$",
        r"$\textbf{1}$",
    ],
    size=35,
)
plt.xlabel(
    r"\textbf{Layer of Llama 3.1-8B}",
    fontweight="bold",
    fontsize=35,
    labelpad=10,
)
# plt.xlabel("Llama 3.1-8B layer", fontsize=14, labelpad=10, fontweight="bold")
plt.title(
    r"\textbf{Cosine Similarity} $\mathbf{(\tilde{z}, \circ)}$ (Higher is better)",
    fontsize=45,
    pad=15,
)
ax.legend(
    title=r"$\textbf{Dataset, Concepts}$",
    title_fontsize=23,
    loc=3,
    prop={
        "size": 25,
        "weight": "bold",
    },
)
# Set grid and aesthetics
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig("layerplot.png", dpi=300)
# plt.show()
