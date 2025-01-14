import numpy as np

import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import seaborn as sns

udrs_binary_2 = [
    0.9690717494531893,
    0.964901856859058,
    0.9651893638561091,
    0.9858168276715547,
    0.983048094993364,
    0.9829261639724016,
    0.9909487261786432,
    0.9897532239171332,
    0.9902402562139567,
]

mean_mccs_binary_2 = [
    0.9685572448992008,
    0.9642387291794329,
    0.9639179310101391,
    0.9850805922898378,
    0.9832268791598769,
    0.9831218310090705,
    0.9907978622655473,
    0.9898931980716276,
    0.989969570673862,
]

std_mccs_binary_2 = [
    0.0034486730605266984,
    0.002513108443668445,
    0.0031736497996879406,
    0.001642862109428793,
    0.0011190291779732504,
    0.0011008048948131064,
    0.0009763470499424323,
    0.0014572809893865547,
    0.0014121594868188418,
]

x_values = np.arange(1, len(udrs_binary_2) + 1)

# Set up the plot
plt.figure(figsize=(10, 6))

# Plot udrs_binary_2 as the first line
sns.lineplot(x=x_values, y=udrs_binary_2, label="UDR", marker="o", linewidth=3)

# Plot mean_mccs_binary_2 with std_mccs_binary_2 as error bars
sns.lineplot(
    x=x_values, y=mean_mccs_binary_2, label="MCC", marker="o", linewidth=3
)

# Add error bars for the mean_mccs_binary_2
plt.errorbar(
    x_values,
    mean_mccs_binary_2,
    yerr=std_mccs_binary_2,
    fmt="none",
    capsize=5,
    color="orange",
    linewidth=2,
)
plt.xticks(
    np.arange(0, len(udrs_binary_2) + 1, step=1),
    labels=[
        "0",
        "(0.01, 11)",
        "(0.01, 13)",
        "(0.01, 15)",
        "(0.007, 11)",
        "(0.007, 13)",
        "(0.007, 15)",
        "(0.005, 11)",
        "(0.005, 13)",
        "(0.005, 15)",
    ],
    fontsize=9,
)
x_mark = x_values[6]
y_mark = mean_mccs_binary_2[6]
plt.annotate(
    f"Optimal HPs",
    xy=(x_mark, y_mark),
    xytext=(x_mark + 0.1, y_mark - 0.005),  # Offset text for readability
    arrowprops=dict(facecolor="black", shrink=0.05),
    fontsize=11,
)
# Add labels and title
plt.xlabel(r"$\{\text{primal lr}, \alpha \}$", fontsize=13)
plt.ylabel("UDR/MCC", fontsize=13)
plt.legend(fontsize=12)
plt.tick_params(width=2)  # Increase the width of the ticks
plt.gca().spines["top"].set_linewidth(2)
plt.gca().spines["right"].set_linewidth(2)
plt.gca().spines["left"].set_linewidth(2)
plt.gca().spines["bottom"].set_linewidth(2)

# Show the plot
plt.savefig("udr_bin2.png")
