#!/usr/bin/env python3
"""
Generate YAML trajectories for two agents that eventually converge
to either cooperation or defection.

YAML schema
-----------
- prompt: 0
cooperation:
    - episode: 0
        - "(D, C)_0"
        - "(C, C)_1"
        ...
    - episode: 1
        - "(C, D)_0"
        - "(C, C)_1"
        ...
defection:
    - episode: 0
        - "(D, C)_0"
        - "(D, D)_1"
        ...
    - episode: 1
        - "(C, D)_0"
        - "(D, D)_1"
        ...
- prompt: N
[...]

Usage (default N=5, T=10):
    python generate_ipd_trajectories.py               # writes trajectories.yaml
    python generate_ipd_trajectories.py 20 15 output.yaml
"""

import sys
import random
import yaml
from typing import List


# -------------------- configuration helpers -------------------- #
def parse_args() -> tuple[int, int, int, str]:
    """Read N, T, and output filename from command-line if provided."""
    P = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    N = int(sys.argv[2]) if len(sys.argv) > 2 else 5  # episodes per prompt
    T = (
        int(sys.argv[2]) if len(sys.argv) > 3 else 10
    )  # timesteps per trajectory
    out_file = sys.argv[3] if len(sys.argv) > 4 else "trajectories.yaml"
    if T < 2:
        raise ValueError("T should be at least 2 to observe convergence.")
    return P, N, T, out_file


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


def make_dataset(P: int, K: int, T: int) -> List[dict]:
    """
    Build the full list-of-prompts structure.
    """
    dataset: List[dict] = []
    for p in range(P):
        block = {
            "prompt": p,
            "cooperation": [],
            "defection": [],
        }
        for i in range(K):
            block["cooperation"].append(
                {
                    "Episode: ": i,
                    "Trajectory of actions by A1 and A2: ": generate_trajectory(
                        T, "C"
                    ),
                }
            )
            block["defection"].append(
                {
                    "Episode: ": i,
                    "Trajectory of actions by A1 and A2: ": generate_trajectory(
                        T, "D"
                    ),
                }
            )
        dataset.append(block)
    return dataset


# -------------------- main entry point -------------------- #
def main() -> None:
    P, N, T, out_path = parse_args()
    data = make_dataset(P, N, T)

    with open(out_path, "w") as f:
        yaml.safe_dump(data, f, sort_keys=False)

    print(
        f"Wrote {P} prompt blocks (each with {N} cooperative and {N} defective "
        f"trajectories of length {T}) to '{out_path}'."
    )


if __name__ == "__main__":
    main()
