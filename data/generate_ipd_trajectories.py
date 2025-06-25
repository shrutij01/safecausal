#!/usr/bin/env python3
"""
Generate YAML trajectories for two agents that eventually converge
to either cooperation or defection.

Usage (default N=5, T=10):
    python make_trajectories.py               # writes trajectories.yaml
    python make_trajectories.py 20 15 output.yaml
"""

import sys
import random
import yaml


# -------------------- configuration helpers -------------------- #
def parse_args() -> tuple[int, int, str]:
    """Read N, T, and output filename from command-line if provided."""
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 5  # trajectories per class
    T = (
        int(sys.argv[2]) if len(sys.argv) > 2 else 10
    )  # timesteps per trajectory
    out_file = sys.argv[3] if len(sys.argv) > 3 else "trajectories.yaml"
    if T < 2:
        raise ValueError("T should be at least 2 to observe convergence.")
    return N, T, out_file


# -------------------- trajectory generation -------------------- #
ACTIONS = ["C", "D"]


def generate_trajectory(T: int, converge_to: str) -> list[str]:
    """
    Produce one trajectory of length T that ends in the pair (converge_to, converge_to)
    for at least the final half of the horizon.

    Example (T=8, converge_to='C'):
        (D,C)_0, (C,D)_1, (C,C)_2, (C,C)_3, (C,C)_4, (C,C)_5, (C,C)_6, (C,C)_7
    """
    assert converge_to in {"C", "D"}
    traj = []

    # Up to halfway -- allow some noise / exploration
    pivot = 2 * T // 3
    for t in range(pivot):
        a1 = random.choice(ACTIONS)
        a2 = random.choice(ACTIONS)
        traj.append(f"({a1}, {a2})_{t}")

    # From pivot onward, lock into the target behaviour
    for t in range(pivot, T):
        traj.append(f"({converge_to}, {converge_to})_{t}")

    return traj


def make_dataset(N: int, T: int) -> dict:
    """Return a dict ready to be dumped with yaml.safe_dump."""
    data = {
        "cooperation": [
            {"id": i, "steps": generate_trajectory(T, converge_to="C")}
            for i in range(N)
        ],
        "defection": [
            {"id": i, "steps": generate_trajectory(T, converge_to="D")}
            for i in range(N)
        ],
    }
    return data


# -------------------- main entry point -------------------- #
def main() -> None:
    N, T, out_file = parse_args()
    dataset = make_dataset(N, T)

    with open(out_file, "w") as f:
        yaml.safe_dump(dataset, f, sort_keys=False)

    print(
        f"Wrote {2*N} trajectories ({N} cooperative, {N} defective) "
        f"to '{out_file}'."
    )


if __name__ == "__main__":
    main()
