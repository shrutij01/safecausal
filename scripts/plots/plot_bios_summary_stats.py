import matplotlib.pyplot as plt
import numpy as np

# -----------------------------
# Data
# -----------------------------
ssae = {
    "strengths": {
        0.0: {"male": 13, "female": 1, "neutral": 22},
        0.5: {"male": 13, "female": 7, "neutral": 16},
        1.0: {"male": 7, "female": 25, "neutral": 4},
        2.0: {"male": 0, "female": 36, "neutral": 0},
        5.0: {"male": 1, "female": 31, "neutral": 4},
    }
}

gemma = {
    "strengths": {
        0.0: {"male": 13, "female": 1, "neutral": 22},
        0.5: {"male": 13, "female": 3, "neutral": 20},
        1.0: {"male": 12, "female": 7, "neutral": 17},
        2.0: {"male": 9, "female": 24, "neutral": 3},
        5.0: {"male": 3, "female": 30, "neutral": 2},
    }
}

strengths = [0.0, 0.5, 1.0, 2.0, 5.0]
x = np.arange(len(strengths))


def normalize(model):
    female, nonfemale = [], []
    for s in strengths:
        f = model["strengths"][s]["female"]
        m = model["strengths"][s]["male"]
        n = model["strengths"][s]["neutral"]
        total = f + m + n
        female.append(f / total)
        nonfemale.append((m + n) / total)
    return np.array(female), np.array(nonfemale)


ssae_f, ssae_nf = normalize(ssae)
gemma_f, gemma_nf = normalize(gemma)

plt.rcParams.update(
    {
        "font.size": 18,
        "axes.titlesize": 20,
        "axes.labelsize": 20,
        "xtick.labelsize": 18,
        "ytick.labelsize": 18,
        "legend.fontsize": 18,
        "axes.labelweight": "bold",
        "axes.titleweight": "bold",
    }
)

# Colorblind-safe, non-competing
FEMALE_COLOR = "#3b4cc0"  # deep indigo
NONFEMALE_COLOR = "#b0b0b0"  # neutral gray

bar_width = 0.75

# -----------------------------
# Figure: stacked bars
# -----------------------------
fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True)

# ---- SSAE ----
ax = axes[0]
ax.bar(x, ssae_f, width=bar_width, color=FEMALE_COLOR, label="Female")
ax.bar(
    x,
    ssae_nf,
    bottom=ssae_f,
    width=bar_width,
    color=NONFEMALE_COLOR,
    label="Male + Neutral",
)

ax.set_title("SSAE")
ax.set_ylabel("Proportion of generations")
ax.set_ylim(0, 1.0)
ax.grid(axis="y", linestyle="--", alpha=0.4)
ax.legend(frameon=False, loc="upper left")

# ---- GemmaScope2B ----
ax = axes[1]
ax.bar(x, gemma_f, width=bar_width, color=FEMALE_COLOR)
ax.bar(x, gemma_nf, bottom=gemma_f, width=bar_width, color=NONFEMALE_COLOR)

ax.set_title("GemmaScope2B")
ax.set_ylabel("Proportion of generations")
ax.set_ylim(0, 1.0)
ax.grid(axis="y", linestyle="--", alpha=0.4)

# -----------------------------
# Shared x-axis
# -----------------------------
axes[1].set_xticks(x)
axes[1].set_xticklabels([str(s) for s in strengths])
axes[1].set_xlabel("Steering strength")

# -----------------------------
# Suptitle
# -----------------------------
fig.suptitle(
    "Bias in Bios: Number of Gender Indicators for Steering Strength",
    fontsize=22,
    fontweight="bold",
    y=0.98,
)

plt.tight_layout(rect=[0, 0, 1, 0.95])
# plt.savefig("bios_gender_transition_bars.png", dpi=300, bbox_inches="tight")
plt.show()
