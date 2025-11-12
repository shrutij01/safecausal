import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams.update({"text.usetex": True, "font.family": "serif"})

ssae_sycophancy_pythia = np.array([0.8750, 0.8621, 0.8699, 0.8725, 0.8702])

ssae_refusal_pythia = np.array([0.8901, 0.8890, 0.8699, 0.8758, 0.9001])

ssae_biasinbios_pythia = np.array([0.9050, 0.8621, 0.8599, 0.8625, 0.8702])

pythia_sycophancy = 0.7432
pythia_refusal = 0.7723
pythia_biasinbios = 0.7695

gemma_sycophancy = 0.8112
gemma_refusal = 0.8520
gemma_biasinbios = 0.8410


ssae_sycophancy_gemma = np.array([0.8521, 0.8350, 0.8499, 0.8400, 0.8402])
ssae_refusal_gemma = np.array([0.8621, 0.8690, 0.8699, 0.8704, 0.8601])
ssae_biasinbios_gemma = np.array([0.8510, 0.8555, 0.8611, 0.8690, 0.8580])


# ----------------------------
# Helper: compute mean and std error
# ----------------------------
def mean_and_se(arr):
    return arr.mean(), arr.std(ddof=1) / np.sqrt(len(arr))


# ----------------------------
# Matplotlib style settings
# ----------------------------
plt.rcParams.update(
    {
        "font.size": 14,  # base font size
        "axes.titlesize": 16,  # title size
        "axes.labelsize": 16,  # x and y labels
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 14,
        "lines.linewidth": 2.5,  # default line width
        "lines.markersize": 8,
    }
)

# ----------------------------
# Dataset labels (LaTeX small caps)
# ----------------------------
datasets = [
    r"\textsc{Sycophancy}",
    r"\textsc{Refusal}",
    r"\textsc{Bias in Bios}",
]

# ----------------------------
# Plot for Pythia
# ----------------------------
pythia_means, pythia_se = [], []
pythia_baseline = [pythia_sycophancy, pythia_refusal, pythia_biasinbios]

for arr in [
    ssae_sycophancy_pythia,
    ssae_refusal_pythia,
    ssae_biasinbios_pythia,
]:
    m, se = mean_and_se(arr)
    pythia_means.append(m)
    pythia_se.append(se)

plt.figure(figsize=(5, 3))
plt.errorbar(
    datasets,
    pythia_means,
    yerr=pythia_se,
    fmt="o-",
    capsize=6,
    color="purple",
    label="SSAE",
)
plt.plot(
    datasets,
    pythia_baseline,
    "r--",
    color="blue",
    label="PythiaSAE70m",
)
plt.title(r"$\mathrm{MCC}_{C}$ (Higher is better)")

plt.ylabel(r"$\mathrm{MCC}_{C}$")
plt.ylim(0.7, 0.95)
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.savefig("pythia_mcc_c.png", dpi=300)

# ----------------------------
# Plot for Gemma
# ----------------------------
gemma_means, gemma_se = [], []
gemma_baseline = [gemma_sycophancy, gemma_refusal, gemma_biasinbios]

for arr in [ssae_sycophancy_gemma, ssae_refusal_gemma, ssae_biasinbios_gemma]:
    m, se = mean_and_se(arr)
    gemma_means.append(m)
    gemma_se.append(se)

plt.figure(figsize=(5, 3))
plt.errorbar(
    datasets,
    gemma_means,
    yerr=gemma_se,
    fmt="o-",
    capsize=6,
    color="purple",
    label="SSAE",
)
plt.plot(datasets, gemma_baseline, "--", color="blue", label="GemmaScope2B")
plt.title(r"$\mathrm{MCC}_{C}$ (Higher is better)")

plt.ylabel(r"$\mathrm{MCC}_{C}$")
plt.ylim(0.7, 0.95)
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.savefig("gemma_mcc_c.png", dpi=300)
