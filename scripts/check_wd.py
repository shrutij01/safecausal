import torch
from psp.linear_sae import LinearSAE, TopK

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
                args.model_dir, "sparse_dict_model", "sparse_dict_model.pth"
            )
            model_dict = torch.load(model_file)
            model.load_state_dict(model_dict)
            wd.append(numpify(model.decoder.weight.data))
    import ipdb

    ipdb.set_trace()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "model_dir",
        nargs="+",
    )
