import torch
from psp.linear_sae import LinearSAE, TopK
import numpy as np

import argparse
import os

from box import Box
import yaml
from psp.data_utils import tensorify, numpify

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def main(args):
    wd = []
    for modeldir in args.model_dir:
        model_config_file = os.path.join(modeldir, "models_config.yaml")
        with open(model_config_file, "r") as file:
            model_config = Box(yaml.safe_load(file))
            topk = TopK(k=int(model_config.k))
            model = LinearSAE(
                input_dim=4096,
                rep_dim=4096,
                normalize=True,
                activation=topk,
            ).to(device)
            model_file = os.path.join(
                modeldir, "sparse_dict_model", "sparse_dict_model.pth"
            )
            model_dict = torch.load(model_file)
            model.load_state_dict(model_dict)
            wd.append(numpify(model.decoder.weight.data))

    def check_equal_arrays(arrays):
        n = len(arrays)
        for i in range(n):
            for j in range(i + 1, n):
                if np.array_equal(arrays[i], arrays[j]):
                    print(f"Array at index {i} is equal to array at index {j}")
                    return True
        print("none of the arrays are equal")
        return False

    check_equal_arrays(wd)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "model_dir",
        nargs="+",
    )
    args = parser.parse_args()
    main(args)
