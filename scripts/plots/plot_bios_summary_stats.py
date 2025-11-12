import matplotlib.pyplot as plt
import numpy as np

# Data
ssae = {
    "baseline": {"male": 13, "female": 1, "neutral": 22},
    "strengths": {
        0.0: {"male": 13, "female": 1, "neutral": 22},
        0.5: {"male": 13, "female": 7, "neutral": 16},
        1.0: {"male": 7, "female": 25, "neutral": 4},
        2.0: {"male": 0, "female": 36, "neutral": 0},
        5.0: {"male": 1, "female": 31, "neutral": 4},
    },
}

gemma = {
    "baseline": {"male": 13, "female": 1, "neutral": 22},
    "strengths": {
        0.0: {"male": 13, "female": 1, "neutral": 22},
        0.5: {"male": 13, "female": 3, "neutral": 20},
        1.0: {"male": 12, "female": 7, "neutral": 17},
        2.0: {"male": 9, "female": 24, "neutral": 3},
        5.0: {"male": 3, "female": 30, "neutral": 2},
    },
}

strengths = [0.0, 0.5, 1.0, 2.0, 5.0]


def extract_counts(model, category):
    return [model["strengths"][s][category] for s in strengths]


# Extract series
ssae_male = extract_counts(ssae, "male")
ssae_female = extract_counts(ssae, "female")
ssae_neutral = extract_counts(ssae, "neutral")

gemma_male = extract_counts(gemma, "male")
gemma_female = extract_counts(gemma, "female")
gemma_neutral = extract_counts(gemma, "neutral")

# Bigger bold fonts
plt.rcParams.update(
    {
        "font.size": 22,
        "axes.titlesize": 26,
        "axes.labelsize": 24,
        "xtick.labelsize": 22,
        "ytick.labelsize": 22,
        "legend.fontsize": 23,
        "font.weight": "bold",
        "axes.labelweight": "bold",
        "axes.titleweight": "bold",
    }
)

fig, ax = plt.subplots(figsize=(18, 11))

# SSAE Female (bright magenta, solid, thick)
ax.plot(
    strengths,
    ssae_female,
    color="#ff00ff",
    linestyle="-",
    marker="o",
    markersize=12,
    linewidth=5,
    label=r"$\mathbf{Female;\ SSAE}$",
)

# SSAE Male (dark purple, dashed, faded)
ax.plot(
    strengths,
    ssae_male,
    color="purple",
    linestyle="--",
    marker="s",
    markersize=10,
    linewidth=4,
    alpha=0.6,
    label=r"$\mathbf{Male;\ SSAE}$",
)

# SSAE Neutral (dark gray, dotted, faded)
ax.plot(
    strengths,
    ssae_neutral,
    color="gray",
    linestyle=":",
    marker="^",
    markersize=10,
    linewidth=4,
    alpha=0.6,
    label=r"$\mathbf{Neutral;\ SSAE}$",
)

# Gemma Female (bright cyan, solid, thick)
ax.plot(
    strengths,
    gemma_female,
    color="#00ffff",
    linestyle="-",
    marker="o",
    markersize=12,
    linewidth=5,
    label=r"$\mathbf{Female;\ GemmaScope2B}$",
)

# Gemma Male (dark blue, dashed, faded)
ax.plot(
    strengths,
    gemma_male,
    color="blue",
    linestyle="--",
    marker="s",
    markersize=10,
    linewidth=4,
    alpha=0.6,
    label=r"$\mathbf{Male;\ GemmaScope2B}$",
)

# Gemma Neutral (black dotted, faded)
ax.plot(
    strengths,
    gemma_neutral,
    color="black",
    linestyle=":",
    marker="^",
    markersize=10,
    linewidth=4,
    alpha=0.6,
    label=r"$\mathbf{Neutral;\ GemmaScope2B}$",
)

# Labels & Title
ax.set_xticks(strengths)
ax.set_xlabel(r"$\mathbf{Steering\ Strength}$")
ax.set_ylabel(r"$\mathbf{Counts\ of\ Female\ vs.\ Male\ Generations}$")
ax.set_title(
    r"$\mathbf{Bias\ in\ Bios:\ Gender\ Indicators\ vs.\ Steering\ Strength}$"
)

ax.legend(ncol=2, frameon=False)
ax.grid(True, linestyle="--", alpha=0.6)

plt.tight_layout()
plt.savefig("bios_summary_stats_bold.png", dpi=300)
# plt.show()
